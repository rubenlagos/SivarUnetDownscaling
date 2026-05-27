# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np 
import random 
import pickle
import os
import json 

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime

from Unet_model import red_unet as unet_arq
from Experimento_5 import (
    composite_wind_loss_with_grad,
    add_sample_weight_from_coast,
    add_unit_weight,
    LrLogger
)

###### Paths ######

#Factors:
scale_factors_path = "Extra/minmax_scales.json"
#data:
path_tensores = "New_datos/" 
#saving experiments
folder_experiments= "experimentos"

with open(f'{scale_factors_path}') as f:
    scales = json.load(f)
U10_MIN, U10_MAX = scales["u10_target"]["min"], scales["u10_target"]["max"]
V10_MIN, V10_MAX = scales["v10_target"]["min"], scales["v10_target"]["max"]

def desnormalize_u10_v10(arr):
    """
    arr: np.ndarray con shape (..., 2) donde arr[...,0]=U10_norm, arr[...,1]=V10_norm
    Normalización esperada: x_norm = (x - min) / (max - min)
    """
    arr = np.asarray(arr)
    assert arr.shape[-1] == 2, "Se espera última dimensión=2 (U10, V10)."
    out = np.empty_like(arr, dtype=np.float32)

    # U10
    out[..., 0] = arr[..., 0] * (U10_MAX - U10_MIN) + U10_MIN
    # V10
    out[..., 1] = arr[..., 1] * (V10_MAX - V10_MIN) + V10_MIN
    return out


#############################################
##### 0. Implementaciones previas ###########
#############################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #verbose

# Fijar las semillas para reproducibilidad
os.environ['PYTHONHASHSEED'] = '42' # A veces, hay fuentes de aleatoriedad en hardware y compiladores que pueden influir
random.seed(42) #controla funciones aleatorias de Python como random
np.random.seed(42) #muchos calculos de redes neuronales utilizan NumPy asi que tambien lo fijamos
tf.random.set_seed(42) #controla operaciones aleatorias dentro de TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Opcional: para garantizar determinismo en operaciones GPU, puede mermar el rendimiento. 

#Estrategia de entrenamiento
tf.debugging.set_log_device_placement(False)
gpus = tf.config.list_logical_devices('GPU')
mirrored_strategy=tf.distribute.MultiWorkerMirroredStrategy()


############################################################
######### 1. Importación de las variables ##################
############################################################

with tf.device('/cpu:0'):

   var_names=  ["u10_input",  "v10_input",  "speed_input","dir_input", "dir_cos_input", "dir_sin_input",
                "psfc_input", "pblh_input", "th2_input",  "t2_input", 

                "hgt_target",  "xland_target", "coastmask_target", "diff_hgt_input", "znt_target",
                "aspect_target", "aspect_sin_target", "aspect_cos_target", "slope_target" , 
                
                "Eplus_input", "Emin_input", "EplusU_input", "EplusV_input",   "EminU_input", "EminV_input",   "EplusUV_input", "EminUV_input", 
                "Utan_input", "Vtan_input",
                "u10_target", "v10_target"]
                #,
                  
           
   variables_tr=[] 
   for name in var_names:  
      tensor_var= tf.convert_to_tensor(np.load(f"{path_tensores}TR/MinMax/{name}_norm_TR.npy"), dtype=tf.float32) 
      variables_tr.append(tensor_var)
      tensor_var=None

   inputs_tr = tf.concat(variables_tr[:(len(var_names)-2)], axis=3)
   target_tr = tf.concat(variables_tr[(len(var_names)-2):len(var_names)], axis=3)
   variables_tr=None 
   print("inputs_tr shape", inputs_tr.shape)
   print("target_tr shape", target_tr.shape)

   variables_va=[]
   for name in var_names:
      tensor_var= tf.convert_to_tensor(np.load(f"{path_tensores}VA/MinMax/{name}_norm_VA.npy"), dtype=tf.float32)
      variables_va.append(tensor_var)
      tensor_var= None

   inputs_va   = tf.concat(variables_va[:(len(var_names)-2)], axis=3)
   target_va   = tf.concat( variables_va[(len(var_names)-2):len(var_names)], axis=3)
   variables_va= None 
   print("inputs_va shape", inputs_va.shape)
   print("target_va shape", target_va.shape)

   BATCH_SIZE = 128
   AUTOTUNE = tf.data.AUTOTUNE

   dataset_train = tf.data.Dataset.from_tensor_slices((inputs_tr, target_tr))
   dataset_val   = tf.data.Dataset.from_tensor_slices((inputs_va, target_va))

   dataset_train = dataset_train.shuffle(buffer_size=len(inputs_tr), reshuffle_each_iteration=True).cache()
   dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=False)
   dataset_val   = dataset_val.batch(BATCH_SIZE, drop_remainder=False)

    # alpha1 default: 0.7
   dataset_train = dataset_train.map(
        add_sample_weight_from_coast(alpha1=1.35, coast_channel_idx=12, threshold=0.5),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    # En validación, usa peso 1 para no sesgar val_loss (o aplica el mismo peso si quieres ver val_loss ponderada).
   dataset_val = dataset_val.map(
        add_unit_weight(),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

   #tensor_X = tf.concat([inputs_tr, inputs_va], axis=0)  
   #tensor_Y = tf.concat([target_tr, target_va], axis=0)  


for batch in dataset_train.take(1):
    print("train batch type:", type(batch))
    print("train batch len:", len(batch))
    x, y, w = batch
    print("x:", x.shape, x.dtype)
    print("y:", y.shape, y.dtype)
    print("w:", w.shape, w.dtype, "min/max:", tf.reduce_min(w).numpy(), tf.reduce_max(w).numpy())

for batch in dataset_val.take(1):
    x, y, w = batch
    print("VAL w:", w.shape, "min/max:", tf.reduce_min(w).numpy(), tf.reduce_max(w).numpy())


######################################################
############## 2. Preparativos red ###################
######################################################

########## 2.1 Hiperparámetros #######################

###### Función de pérdida #################
num_clases = 2  #No cambiar
factor = 1
loss_name = "loss_exp5_alpha135_3blocks"
alpha=1.25

##### Modelo ##############################
epochs = 1
lr_actual = float(model.optimizer.learning_rate.numpy()) if 'model' in locals() else 0.001  # o tu LR
input_channels = len(var_names) - 2

##### 2.2 Callbacks 
lr_logger = LrLogger()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,  # Aumenta según la frecuencia de tus fluctuaciones
    min_delta=0.00075,  # Umbral de cambio mínimo en la pérdida para que cuente como mejora originalmente 0.001
    restore_best_weights=True  # Restaura el mejor modelo encontrado antes del punto de early stopping
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reducción del learning rate (ajusta según el comportamiento de tu modelo)
    patience=8,  # Número de épocas sin mejora antes de reducir el learning rate
    min_lr=1e-6  # Learning rate mínimo
)


##### 2.3 Implementación en gpu
# Hiperparámetros iniciales recomendados Exp5
LAM_UV  = 1.0
LAM_MAG = 0.5 #def 0.5
LAM_ANG = 0.3 #def 0.3
LAM_GRAD = 0.25
HUBER_DELTA = 0.1

loss_exp5 = composite_wind_loss_with_grad(
    lam_uv=LAM_UV,
    lam_mag=LAM_MAG,
    lam_ang=LAM_ANG,
    lam_grad=LAM_GRAD,
    huber_delta=HUBER_DELTA
)

# Importante: ajusta input_size al número real de canales de entrada que usas
# Aquí sólo muestro el patrón, mantén tu valor correcto en input_size


with mirrored_strategy.scope():
   #model = red_unet(grad_loss=custom_loss(factor)) #default input_size (32,32,4) y num_clases=2 
    model = unet_arq(
        input_size=(32, 32, len(var_names)-2),  # pon aquí tu C_in actual
        grad_loss=loss_exp5,
        num_clases=2
    )
##########################################
########### 3. Entrenamiento red #########
##########################################


history = model.fit(
    dataset_train,
    batch_size=BATCH_SIZE, 
    epochs=epochs, 
    validation_data=(dataset_val),
    callbacks=[early_stopping, reduce_lr, lr_logger])

epochs_trained = len(history.history["loss"])
print("Epocas entrenadas efectivas:", epochs_trained)

print("[2.3] Entrenamiento red finalizado correctamente")



###########################################
########### 4. Loggeo Modelo ##############
###########################################


### 4.1 Creación de carpetas
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
exp_name = f"Experimento_5_UNET_ep{epochs}_epTR{epochs_trained}_bs{BATCH_SIZE}_lr{lr_actual:.4g}_f{factor}_{loss_name}_ch{input_channels}_{timestamp}"

base_dir = os.path.join(folder_experiments, exp_name)
paths = {
    "root": base_dir,
    "model_dir": os.path.join(base_dir, "modelo"),
    "weights_dir": os.path.join(base_dir, "pesos"),
    "history_dir": os.path.join(base_dir, "historial"),
    "preds_dir": os.path.join(base_dir, "predicciones"),
    "meta_dir": os.path.join(base_dir, "meta"),
}
for p in paths.values():
    os.makedirs(p, exist_ok=True)

### 4.2 Metadata 
metadata = {
    "epochs": epochs,
    "epochs_trained": epochs_trained,
    "batch_size": BATCH_SIZE,
    "learning_rate": lr_actual,
    "factor": factor,
    "alpha": alpha,
    "num_clases": num_clases,
    "loss": loss_name,
    "input_size": [32, 32, input_channels],
    "timestamp": timestamp,
    "var_list": var_names
}
with open(os.path.join(paths["meta_dir"], "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)


### 4.3 guardado de modelo 
model.save(os.path.join(paths["model_dir"], "modelo.keras"))
model.save(os.path.join(paths["model_dir"], "saved_model_tf"), save_format="tf")
model.save_weights(os.path.join(paths["weights_dir"], "pesos.h5"))
with open(os.path.join(paths["history_dir"], "history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

print("[4.3] El modelo ha sido guardado correctamente")


### 4.4 Emisión de predicciónes

def guardar_preds(X, split_name, batch=32):
    preds = model.predict(X, batch_size=batch, verbose=1)

    # Si tus targets/predicciones son 2 canales U/V ya normalizados con min-max:
    preds_dn  = desnormalize_u10_v10(preds)
    np.save(os.path.join(paths["preds_dir"], f"y_pred_{split_name}.npy"), preds_dn)

guardar_preds(dataset_val, "val", batch=BATCH_SIZE)

print(f"Experimento guardado en: {base_dir}")