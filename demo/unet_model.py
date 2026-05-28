# Experiment 4 — Loss Function: Vectorial + Magnitude + Angular (Wind)
#
# Objective:
# - Replaces the current loss with a composite loss function specifically designed for wind
#   vector fields U, V, targeting improvements in both direction and magnitude accuracy:
#   MAE(U,V) + MAE(|V|) + angular loss.
#
# What is tested:
# - Loss function components:
#   * L_UV:  MAE over U and V components (per-component error).
#   * L_mag: MAE over wind speed magnitude (|V|).
#   * L_ang: Angular loss = 1 − cos(Δθ) = 1 − dot(Vp,Vt) / (||Vp||·||Vt|| + ε).
# - Initial weighting scheme: L = λ1·L_UV + λ3·L_mag + λ4·L_ang,
#   with λ1 = 1.0, λ3 = 0.5, λ4 = 0.3.
#
# Motivation:
# - Optimizing solely over U and V components may bias the model toward minimizing
#   projection error while neglecting directional accuracy. The inclusion of magnitude
#   and angular terms — established in optical flow and wind field literature — is
#   expected to reduce angular error and improve physical-vectorial coherence.
#
# Hypothesis:
# - A reduction in angular error and RMSE(speed) is anticipated, particularly in cases
#   involving complex topography (cases 3, 7, and 8), with performance exceeding the
#   global baseline.
#
# What remains unchanged:
# - Model architecture (Exp2), optimizer, callbacks, batch size, and data pipeline.
#
# Success criteria:
# - Decrease in global and per-case RMSE(speed); reduction in mean angular error of
#   at least 10–20% relative to the baseline.
#
# Notes:
# - This loss function serves as the foundation for Exp5, where Huber gradient terms
#   and edge-aware weights based on terrain height (HGT) and coastline proximity will
#   be incorporated.


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Concatenate, Lambda, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Multiply

def custom_loss(lambda_factor):
    def loss(y_true, y_pred):
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        dy_true_dx = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        dy_true_dy = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        dy_pred_dx = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        dy_pred_dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        
        grad_diff_x = tf.reduce_mean(tf.abs(dy_true_dx - dy_pred_dx))
        grad_diff_y = tf.reduce_mean(tf.abs(dy_true_dy - dy_pred_dy))
        grad_loss = grad_diff_x + grad_diff_y
        
        total_loss = l1_loss + lambda_factor * grad_loss
        ### Mirar l1_loss y grad_loss por separado
        return total_loss
    
    return loss

def reflect_padding(x, padding_size):
    return tf.pad(x, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='REFLECT')

def reflect_padding_layer(x):
    return reflect_padding(x, 1)

def residual_conv_block(x_in, num_filters, l2_lambda=0.001):
    # Rama principal
    x = Lambda(reflect_padding_layer)(x_in)  # reflect padding igual que tu conv_block
    x = Conv2D(num_filters, (3,3), padding="valid", kernel_regularizer=l2(l2_lambda), use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(reflect_padding_layer)(x)
    x = Conv2D(num_filters, (3,3), padding="valid", kernel_regularizer=l2(l2_lambda), use_bias=True)(x)
    x = BatchNormalization()(x)
    # No activación aquí; la aplicamos después de sumar

    # Atajo (proyección) si cambia el número de canales
    if x_in.shape[-1] != num_filters:
        skip = Conv2D(num_filters, (1,1), padding="same", kernel_regularizer=l2(l2_lambda), use_bias=True)(x_in)
        skip = BatchNormalization()(skip)
    else:
        skip = x_in

    x = Add()([x, skip])
    x = Activation("relu")(x)
    return x

def encoder_block(input, num_filters, size=(2,2)):
    x = residual_conv_block(input, num_filters)
    p = MaxPool2D(size)(x)
    return x, p

def decoder_block_T(input, skip_features, num_filters, use_cbam=True):
    x = Conv2DTranspose(num_filters, (3, 3), strides=(2,2), padding="same")(input)
    if use_cbam:
        skip_features = cbam_block(skip_features, reduction=8)
    x = Concatenate()([x, skip_features])
    x = residual_conv_block(x, num_filters)
    return x

def l1_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# Métrica para el grad_loss
def grad_loss_metric(y_true, y_pred):
    dy_true_dx = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    dy_true_dy = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dy_pred_dx = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    dy_pred_dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    
    grad_diff_x = tf.reduce_mean(tf.abs(dy_true_dx - dy_pred_dx))
    grad_diff_y = tf.reduce_mean(tf.abs(dy_true_dy - dy_pred_dy))
    return grad_diff_x + grad_diff_y



def cbam_block(x, reduction=8):
    # Channel Attention
    ch = x.shape[-1]
    # Global Poolings conservando dims
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    avg_pool = Reshape((1,1,ch))(avg_pool)
    max_pool = Reshape((1,1,ch))(max_pool)
    shared_mlp_1 = Conv2D(ch // max(1, reduction), 1, activation='relu', padding='same')
    shared_mlp_2 = Conv2D(ch, 1, activation='sigmoid', padding='same')
    ca = Add()([shared_mlp_2(shared_mlp_1(avg_pool)),
                shared_mlp_2(shared_mlp_1(max_pool))])
    x_ca = Multiply()([x, ca])

    # Spatial Attention
    avg = tf.reduce_mean(x_ca, axis=-1, keepdims=True)
    mx  = tf.reduce_max(x_ca, axis=-1, keepdims=True)
    sa  = Concatenate(axis=-1)([avg, mx])
    sa  = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(sa)
    x_out = Multiply()([x_ca, sa])
    return x_out


from tensorflow.keras import backend as K
import math

def _compute_speed(u, v, eps=1e-6):
    return tf.sqrt(tf.maximum(u*u + v*v, eps))

def angular_error_deg_metric(y_true, y_pred, eps=1e-6):
    """
    Error angular medio entre (U,V) predicho y verdadero, en grados.
    y_true, y_pred: [..., 2] con canales [U, V] (normalizados en tu pipeline).
    """
    u_t = y_true[..., 0]
    v_t = y_true[..., 1]
    u_p = y_pred[..., 0]
    v_p = y_pred[..., 1]

    dot = u_p*u_t + v_p*v_t
    mag_p = _compute_speed(u_p, v_p, eps)
    mag_t = _compute_speed(u_t, v_t, eps)

    cos_sim = dot / (mag_p * mag_t + eps)
    # Clamp numérico para evitar NaNs por redondeos
    cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)

    ang_rad = tf.acos(cos_sim)          # en radianes
    ang_deg = ang_rad * (180.0 / math.pi)
    return tf.reduce_mean(ang_deg)

def compute_speed(u, v, eps=1e-6):
    # Magnitud del viento |V|
    return tf.sqrt(tf.maximum(u*u + v*v, eps))

def angular_loss_component(u_p, v_p, u_t, v_t, eps=1e-6):
    # 1 - cos(delta) = 1 - dot/||p||/||t||
    dot = u_p*u_t + v_p*v_t
    mag_p = compute_speed(u_p, v_p, eps)
    mag_t = compute_speed(u_t, v_t, eps)
    cos_sim = dot / (mag_p * mag_t + eps)
    # Clamp por seguridad numérica (evitar valores >1 o < -1 por redondeo)
    cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim

def composite_wind_loss(lam_uv=1.0, lam_mag=0.5, lam_ang=0.3, eps=1e-6):
    """
    L = lam_uv * MAE(U,V) + lam_mag * MAE(|V|) + lam_ang * AngularLoss
    y_true, y_pred: [..., 2] (U,V) normalizados 0-1 si ese es tu preprocesamiento.
    """
    def loss(y_true, y_pred):
        u_t = y_true[..., 0]
        v_t = y_true[..., 1]
        u_p = y_pred[..., 0]
        v_p = y_pred[..., 1]

        # MAE en componentes
        l_uv = tf.reduce_mean(tf.abs(u_p - u_t) + tf.abs(v_p - v_t))

        # MAE en magnitud
        sp_t = compute_speed(u_t, v_t, eps)
        sp_p = compute_speed(u_p, v_p, eps)
        l_mag = tf.reduce_mean(tf.abs(sp_p - sp_t))

        # Pérdida angular
        l_ang_map = angular_loss_component(u_p, v_p, u_t, v_t, eps)
        l_ang = tf.reduce_mean(l_ang_map)

        total = lam_uv * l_uv + lam_mag * l_mag + lam_ang * l_ang
        return total
    return loss

def red_unet(input_size=(32,32,20), grad_loss=custom_loss(lambda_factor=1.0), num_clases=2):
    inputs = Input(input_size)

    s1, p1 = encoder_block(inputs, 64, (2,2))   # Bloque 1
    s2, p2 = encoder_block(p1, 128, (2,2))      # Bloque 2
    s3, p3 = encoder_block(p2, 256, (2,2))      # Bloque 3
    #s4, p4 = encoder_block(p3, 512, (2,2))

    b1 = residual_conv_block(p3, 512)

    #d0 = decoder_block_T(b1, s4, 512)  # Bloque 4
    d1 = decoder_block_T(b1, s3, 256)  # Bloque 4
    d2 = decoder_block_T(d1, s2, 128)  # Bloque 3
    d3 = decoder_block_T(d2, s1, 64)   # Bloque 2

    outputs_res = Conv2D(num_clases, (1, 1), activation='linear')(d3)

    model = Model(inputs=[inputs], outputs=[outputs_res])

    model.compile(
        loss=grad_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mse', 'mae', 'mape', angular_error_deg_metric],
        weighted_metrics = ['mse', 'mae', 'mape', angular_error_deg_metric]
    )

    return model