# =============================================================================
# Experiment — Edge-Aware Loss Function with Huber Gradient Term
#
# Objective:
# - Retains the best-performing architecture to date (Exp2: Residual U-Net + CBAM)
#   and modifies exclusively the loss function and spatial weighting scheme, with the
#   aim of reducing prediction errors in coastal zones and fjords.
#
# What is tested:
# - A composite wind-specific loss function:
#   L = λ1·MAE(U,V) + λ3·MAE(|V|) + λ4·L_ang (from Exp4) + λ5·Huber(∇U,∇V) (new term).
# - Spatial weighting via sample_weight derived from the coastline channel (index 11, 0-indexed):
#   w = 1 + α1·I_coast. Applied to per-pixel terms (component MAE, magnitude MAE, and angular
#   loss). The gradient term (L_grad) is left unweighted to avoid amplifying noise at boundaries.
#
# Justification (literature and practice):
# - Edge-aware weighting: widely adopted in super-resolution and remote sensing to prioritize
#   boundaries and critical transitions. In wind downscaling, coastal zones and fjords concentrate
#   the largest errors due to abrupt land-sea contrasts and channeling effects.
# - Vectorial and angular terms: standard in vector field tasks (optical flow, wind fields) for
#   reducing directional error and improving physical coherence; typically yield improvements
#   in RMSE(speed).
# - Huber gradient term: preserves spatial detail by preventing over-smoothing while remaining
#   robust to outliers; commonly employed in super-resolution tasks.
#
# Hypothesis:
# - A reduction in RMSE over coastal zones and fjords is expected, along with a decrease in
#   mean angular error, with overall performance surpassing the global baseline when evaluated
#   over the full domain.
#
# What remains unchanged:
# - Model architecture (Exp2), optimizer, callbacks, batch size (128), and the base data
#   pipeline (sample_weight is the only addition).
#
# Implementation:
# - The following components are provided:
#   * composite_wind_loss_with_grad: Exp4 loss extended with the Huber gradient term.
#   * angular_error_deg_metric:      auxiliary metric reporting mean angular error in degrees.
#   * add_sample_weight_from_coast:  tf.data function that constructs sample_weight from
#                                    channel index 11.
#   * Model compilation example with recommended initial hyperparameters.
#
# Initial hyperparameters:
# - Batch size:  128
# - λ1 = 1.0  (MAE U, V components)
# - λ3 = 0.5  (MAE wind speed magnitude)
# - λ4 = 0.3  (Angular loss)
# - λ5 = 0.25 (Huber gradient term),  δ = 0.1
# - α1 = 0.7  (Coastal weight)
# - Max epochs: 150–200 (EarlyStopping guards against overfitting or stagnation)
#
# Metrics to report:
# - Global and per-case RMSE (consistent with previous experiments)
# - Mean angular error (degrees)
# - Stratified post-processing: RMSE and angular error for coastal vs. interior regions
# =============================================================================

import tensorflow as tf
import math

# -------------------------
# Utilidades de pérdida
# -------------------------

class LrLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        opt = self.model.optimizer
        # Maneja LR fijo o schedules que devuelven un tensor
        lr = opt.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        if tf.is_tensor(lr):
            lr = float(tf.keras.backend.get_value(lr))
        logs['lr'] = lr
        print(f"\n[Epoch {epoch+1}] lr={lr:.6g}")

def _compute_speed(u, v, eps=1e-6):
    return tf.sqrt(tf.maximum(u*u + v*v, eps))

def _angular_loss_map(u_p, v_p, u_t, v_t, eps=1e-6):
    # 1 - cos(delta) = 1 - dot/||p||/||t||
    dot = u_p*u_t + v_p*v_t
    mag_p = _compute_speed(u_p, v_p, eps)
    mag_t = _compute_speed(u_t, v_t, eps)
    cos_sim = dot / (mag_p * mag_t + eps)
    cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim  # mapa [B,H,W]

def _huber(x, delta=0.1):
    absx = tf.abs(x)
    return tf.where(absx <= delta, 0.5*tf.square(x), delta*(absx - 0.5*delta))

def _grad_xy(z):
    """
    Diferencias forward para derivadas espaciales. z shape: [B,H,W,1] o [B,H,W]
    Retorna:
      dx: [B,H,W-1,1]
      dy: [B,H-1,W,1]
    """
    if z.shape.rank == 3:
        z = tf.expand_dims(z, -1)
    dx = z[:, :, 1:, :] - z[:, :, :-1, :]
    dy = z[:, 1:, :, :] - z[:, :-1, :, :]
    return dx, dy

def composite_wind_loss_with_grad(lam_uv=1.0, lam_mag=0.5, lam_ang=0.3, lam_grad=0.25, huber_delta=0.1, eps=1e-6):
    """
    L_total = lam_uv * MAE(U,V) + lam_mag * MAE(|V|) + lam_ang * Angular + lam_grad * Huber(∇U,∇V)
    - Compatible con sample_weight espacial (Keras lo aplica automáticamente a los términos por-píxel).
    - y_true, y_pred: [..., 2] (U,V) normalizados (tu pipeline).
    """
    def loss(y_true, y_pred):
        u_t = y_true[..., 0]; v_t = y_true[..., 1]
        u_p = y_pred[..., 0]; v_p = y_pred[..., 1]

        # Términos por-píxel (Keras aplicará sample_weight si se suministra)
        l_uv_map  = tf.abs(u_p - u_t) + tf.abs(v_p - v_t)
        sp_t = _compute_speed(u_t, v_t, eps)
        sp_p = _compute_speed(u_p, v_p, eps)
        l_mag_map = tf.abs(sp_p - sp_t)
        l_ang_map = _angular_loss_map(u_p, v_p, u_t, v_t, eps)

        # Término de gradiente Huber (no usar sample_weight para no amplificar ruido en bordes)
        up = tf.expand_dims(u_p, -1); ut = tf.expand_dims(u_t, -1)
        vp = tf.expand_dims(v_p, -1); vt = tf.expand_dims(v_t, -1)
        updx, updy = _grad_xy(up); utdx, utdy = _grad_xy(ut)
        vpdx, vpdy = _grad_xy(vp); vtdx, vtdy = _grad_xy(vt)

        l_grad = (
            tf.reduce_mean(_huber(updx - utdx, huber_delta)) +
            tf.reduce_mean(_huber(updy - utdy, huber_delta)) +
            tf.reduce_mean(_huber(vpdx - vtdx, huber_delta)) +
            tf.reduce_mean(_huber(vpdy - vtdy, huber_delta))
        )

        # Reducir por promedio (Keras integrará sample_weight cuando corresponda)
        l_uv  = tf.reduce_mean(l_uv_map)
        l_mag = tf.reduce_mean(l_mag_map)
        l_ang = tf.reduce_mean(l_ang_map)

        total = lam_uv*l_uv + lam_mag*l_mag + lam_ang*l_ang + lam_grad*l_grad
        return total

    return loss

# -------------------------
# Métrica auxiliar
# -------------------------
def angular_error_deg_metric(y_true, y_pred, eps=1e-6):
    """
    Error angular medio entre (U,V) predicho y verdadero, en grados.
    """
    u_t = y_true[..., 0]; v_t = y_true[..., 1]
    u_p = y_pred[..., 0]; v_p = y_pred[..., 1]
    dot = u_p*u_t + v_p*v_t
    mag_p = _compute_speed(u_p, v_p, eps)
    mag_t = _compute_speed(u_t, v_t, eps)
    cos_sim = dot / (mag_p * mag_t + eps)
    cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
    ang_rad = tf.acos(cos_sim)
    ang_deg = ang_rad * (180.0 / math.pi)
    return tf.reduce_mean(ang_deg)

# -------------------------
# tf.data: sample_weight desde canal "costa" (índice 11)
# -------------------------
def add_sample_weight_from_coast(alpha1=0.7, coast_channel_idx=12, threshold=0.5):
    """
    Para usar con dataset.map: (x, y) -> (x, y, w)
    - x: [B,H,W,C], y: [B,H,W,2], w: [B,H,W,1]
    - w = 1 + alpha1 * I_costa, donde I_costa = bin(x[..., coast_channel_idx] >= threshold)
    """
    @tf.function
    def _map(x, y):
        coast = x[..., coast_channel_idx:coast_channel_idx+1]  # [B,H,W,1]
        if threshold is not None:
            coast = tf.cast(coast >= threshold, x.dtype)
        w = 1.0 + alpha1 * coast
        return (x, y, w)
    return _map

def add_unit_weight():
    """
    Para validación si no quieres ponderar la val_loss:
    (x, y) -> (x, y, w=1)
    """
    @tf.function
    def _map(x, y):
        w = tf.ones_like(y[..., :1])
        return (x, y, w)
    return _map

# -------------------------
# Ejemplo de integración (compile)
# -------------------------
def compile_model_exp5(model,
                       lam_uv=1.0, lam_mag=0.5, lam_ang=0.3, lam_grad=0.25,
                       huber_delta=0.1, lr=1e-3):
    """
    Compila el modelo (arquitectura de Exp2) con la pérdida de Exp5 y métricas.
    """
    loss_fn = composite_wind_loss_with_grad(
        lam_uv=lam_uv,
        lam_mag=lam_mag,
        lam_ang=lam_ang,
        lam_grad=lam_grad,
        huber_delta=huber_delta
    )

    model.compile(
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            # Mantén aquí tus métricas previas si las tenías (mse/mae/mape custom)
            angular_error_deg_metric  # Métrica auxiliar clave para Exp5
        ]
    )
    return model

