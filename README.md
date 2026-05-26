# 🌬️ U-Net Wind Downscaling for Complex Coastal Terrain in Southern Chile

<p align="center">
  <img src="img/graphical_abstract.jpg" alt="Graphical Abstract" width="850"/>
</p>

> **A deep learning framework for downscaling 3 km WRF forecasts to 333 m near-surface wind fields over southern Chile's fjord-dominated coastal domain.**

---

## 📖 About

Accurate wind forecasting in complex terrain is limited by the coarse resolution of numerical weather prediction (NWP) models. This work presents a **2D-to-2D U-Net** trained as a high-resolution WRF emulator to downscale 10-m wind components (*U*, *V*) from **3 km WRF forecasts** to **~333 m resolution** over a highly complex coastal–insular domain in southern Chile.

The model takes as input:
- 🌀 Bicubically interpolated low-resolution wind fields (*U10*, *V10*)
- 🏔️ 29 high-resolution topographic descriptors (elevation, slope, aspect, flow-deflection indices)

Training is guided by a **physically informed composite loss** that jointly penalizes errors in wind speed, direction, and spatial gradients, combined with **coastal-mask spatial weighting** to prioritize complex terrain regions.

### ✨ Key Results

| Metric | Value |
|---|---|
| Global RMSE reduction | **~18%** vs. bicubic baseline |
| Vector RMSE improvement | **17.75%** |
| Coastal / high-topography zones | **>22%** improvement |
| Spatiotemporal coherence (*r* ≥ 0.8) | **93%** of domain |

> The framework is **operationally deployed** in the SiVAR forecasting system, providing high-resolution wind fields for environmental and maritime applications.

---

## 📁 Repository Structure

