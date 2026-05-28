# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np 
import random 
import pickle
import os
import json 

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime

from utils.unet_model import red_unet as unet_arq
from utils.unet_functions import (
    composite_wind_loss_with_grad,
    add_sample_weight_from_coast,
    add_unit_weight,
    LrLogger
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

###### 0. Some functions #########
def desnormalize_u10_v10(arr):
    """
    Reverse min-max normalization for U10 and V10 wind components.

    Converts normalized values back to their original physical scale using
    the inverse of min-max normalization: x = x_norm * (max - min) + min.

    Parameters
    ----------
    arr : array_like
        Array with shape (..., 2) where arr[..., 0] contains normalized U10
        (eastward wind) values and arr[..., 1] contains normalized V10
        (northward wind) values. Values are expected in the [0, 1] range.

    Returns
    -------
    np.ndarray
        Array of the same shape as the input (dtype float32) with
        denormalized U10 and V10 values in their original units (m/s).

    Raises
    ------
    AssertionError
        If the last dimension of `arr` is not 2.
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

###### 1. Paths ######
path_scalefactors = "../Extra/minmax_scales.json"
path_data= "extra/" 
path_weights= "weights/"

###### 2. Scale Factors ##########
with open(f'{path_scalefactors}') as f:
    scales = json.load(f)
U10_MIN, U10_MAX = scales["u10_target"]["min"], scales["u10_target"]["max"]
V10_MIN, V10_MAX = scales["v10_target"]["min"], scales["v10_target"]["max"]

###### 3. Import Vars ################

# Variable index mapping:
# ┌───────┬─────────────────────┐
# │ Index │ Variable            │
# ├───────┼─────────────────────┤
# │   0   │ u10_input           │
# │   1   │ v10_input           │
# │   2   │ speed_input         │
# │   3   │ dir_input           │
# │   4   │ dir_cos_input       │
# │   5   │ dir_sin_input       │
# │   6   │ psfc_input          │
# │   7   │ pblh_input          │
# │   8   │ th2_input           │
# │   9   │ t2_input            │
# │  10   │ hgt_target          │
# │  11   │ xland_target        │
# │  12   │ coastmask_target    │
# │  13   │ diff_hgt_input      │
# │  14   │ znt_target          │
# │  15   │ aspect_target       │
# │  16   │ aspect_sin_target   │
# │  17   │ aspect_cos_target   │
# │  18   │ slope_target        │
# │  19   │ Eplus_input         │
# │  20   │ Emin_input          │
# │  21   │ EplusU_input        │
# │  22   │ EplusV_input        │
# │  23   │ EminU_input         │
# │  24   │ EminV_input         │
# │  25   │ EplusUV_input       │
# │  26   │ EminUV_input        │
# │  27   │ Utan_input          │
# │  28   │ Vtan_input          │
# └───────┴─────────────────────┘
                
logger.info("Loading data...")
X  = np.load("extra/X_validacion_example.npy").astype(np.float32)
logger.info("Input data shape: %s", X.shape)

###### 4. Model import ################
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE
dataset_val   = tf.data.Dataset.from_tensor_slices((X))
dataset_val   = dataset_val.batch(BATCH_SIZE, drop_remainder=False)
dataset_val = dataset_val.prefetch(AUTOTUNE)

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

model = unet_arq(
    input_size=(32, 32, 29),  # pon aquí tu C_in actual
    grad_loss=loss_exp5,
    num_clases=2
)

logger.info("Loading weights...")
model.load_weights(f'weights/weights.h5')

##### 5. Prediction ############
logger.info("Starting prediction...")
Y_pred          = model.predict(X, batch_size=BATCH_SIZE , verbose=1)
logger.info("Prediction shape: %s", Y_pred.shape)

Y_pred_desnorm  = desnormalize_u10_v10(Y_pred)
Y_pred_map      =  patches_to_maps_batched(Y_pred_desnorm)

##### 6. Optional Saving
pred_name = 'Y_pred_example.npy'
path_pred = path_data + pred_name
np.save(path_pred,Y_pred_map)
logger.info("Predictions saved to: %s", path_pred)

