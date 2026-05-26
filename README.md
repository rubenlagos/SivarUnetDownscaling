# 🌬️ WindNet: CNN-based Wind Downscaling for Complex Terrain

<p align="center">
  <img src="graphical_abstract.jpg" alt="Graphical Abstract" width="900"/>
</p>

> **Hyper-resolution downscaling of near-surface winds over southern Chile's fjord-dominated coastal domain using a physically informed U-Net.**

---

## 📖 About

Accurate wind forecasting in complex terrain is limited by the coarse resolution of numerical weather prediction (NWP) models. **WindNet** addresses this by applying a lightweight **2D-to-2D U-Net** to downscale 10-m wind components (*U*, *V*) from kilometer-scale NWP output to **333-m resolution** across southern Chile.

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

> The framework is **operationally deployed** in the [SiVAR](https://sivar.cl) forecasting system, providing high-resolution wind fields for environmental and maritime applications.

---

## 📁 Repository Structure
