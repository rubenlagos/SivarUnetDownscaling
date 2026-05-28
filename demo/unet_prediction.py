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
from Unet_complements import (
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

def patches_to_maps_batched(patches: np.ndarray) -> np.ndarray:
    """
    Reconstruct full maps from a batch of patches with 2 channels.

    Reassembles a sequence of (32x32) patches into their original
    (384x416) spatial maps, processing multiple maps in a single batch.

    Parameters
    ----------
    patches : np.ndarray
        Array of shape (N, 32, 32, 2) where N = 156 * B,
        B is the batch size and 156 = 12 * 13 patches per map.
        The last dimension contains 2 channels (e.g. U10, V10).

    Returns
    -------
    np.ndarray
        Reconstructed maps of shape (B, 384, 416, 2).

    Raises
    ------
    ValueError
        If patch spatial dimensions are not 32x32, N is not a multiple
        of 156, or the number of channels is not 2.
    """
    PH, PW          = 32, 32
    NY, NX          = 384 // PH, 416 // PW  # 12, 13
    PATCHES_PER_MAP = NY * NX               # 156

    N, ph, pw, C = patches.shape
    if (ph, pw) != (PH, PW):
        raise ValueError(f"Expected patch size 32x32, got {ph}x{pw}.")
    if N % PATCHES_PER_MAP != 0:
        raise ValueError(f"N={N} is not a multiple of {PATCHES_PER_MAP}.")
    if C != 2:
        raise ValueError(f"Expected 2 channels, got {C}.")

    B = N // PATCHES_PER_MAP

    # (N, 32, 32, 2) -> (B, NY, NX, PH, PW, 2) -> (B, NY, PH, NX, PW, 2) -> (B, H, W, 2)
    out = (patches
           .reshape(B, NY, NX, PH, PW, C)
           .transpose(0, 1, 3, 2, 4, 5)
           .reshape(B, NY * PH, NX * PW, C))

    return out  # (B, 384, 416, 2)

exp_name = f"Experimento_5_UNET_ep200_epTR127_bs128_lr0.001_f1_loss_exp5_alpha135_ch29_20251202-074631"

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
                  




   inputs_va   = np.load("Extra/X_validacion_example.npy").astype(np.float32)
   target_va   = np.load("Extra/Y_validacion_example.npy").astype(np.float32)

   print("inputs_va shape", inputs_va.shape)
   print("Shape: ", inputs_va.shape, np.min(inputs_va), '-', np.max(inputs_va))

   BATCH_SIZE = 128
   AUTOTUNE = tf.data.AUTOTUNE

   dataset_val   = tf.data.Dataset.from_tensor_slices((inputs_va))
   dataset_val   = dataset_val.batch(BATCH_SIZE, drop_remainder=False)
   dataset_val = dataset_val.prefetch(AUTOTUNE)


##### 2.3 Implementación en gpu
# Hiperparámetros iniciales recomendados Exp5
LAM_UV  = 1.0
LAM_MAG = 0.5 
LAM_ANG = 0.3 
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
        input_size=(32, 32, 29),  # pon aquí tu C_in actual
        grad_loss=loss_exp5,
        num_clases=2
    )

    model.load_weights(f'experimentos/{exp_name}/pesos/pesos.h5')
    print("HOLA MUNDO")



###########################################
########### 4. Loggeo Modelo ##############
###########################################

preds = model.predict(dataset_val, batch_size=BATCH_SIZE , verbose=1)
preds_dn  = desnormalize_u10_v10(preds)
y_pred =  patches_to_maps_batched(preds_dn)

np.save('Extra/Y_pred_example.npy', y_pred)
np.save('Extra/Y_validation_map_example.npy', patches_to_maps_batched(desnormalize_u10_v10(target_va)))


