# Data Engineering Handoff Document: OceanEmbed

To the ML Team: This document outlines the structure of the preprocessed datasets provided by the Data Engineering pipeline. 

The raw, unaligned NetCDF files from Copernicus Marine Service and NASA PO.DAAC have been programmatically downloaded, cleaned, and merged into a unified PyTorch-ready format.

## Overview

The data pipeline has generated two fully aligned NetCDF files located in the `data/processed/` directory:
1. `processed_surface_data.nc` (Features / Input $X$)
2. `processed_subsurface_data.nc` (Target / Output $Y$)

Both files share exactly the same temporal (`time`) and spatial (`lat`, `lon`) dimensions.

### Grid Dimensions
- **Spatial Bounds**: North Indian Ocean (Lat: 5°N–30°N, Lon: 45°E–105°E)
- **Spatial Resolution**: 0.25° x 0.25° (101 Latitudes x 241 Longitudes)
- **Temporal Resolution**: Daily (`resampled to 1D`)
- **Depth Levels (Y only)**: 15 standard depth layers:
  `[0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0]`

---

## 1. Feature Data ($X$ Tensor)
**File**: `data/processed/processed_surface_data.nc`
**Shape**: `[Time, 7, Lat, Lon]`

This file contains 7 standardized physical variables representing the surface observations:
| Channel Index | Variable Name | Source Mission / Product |
| :---: | :--- | :--- |
| `0` | `sst` (Sea Surface Temperature) | OSTIA L4 Reprocessed |
| `1` | `sss` (Sea Surface Salinity) | SMOS/SMAP MULTIOBS |
| `2` | `ssh` (Sea Surface Height / ADT) | DUACS L4 Altimetry |
| `3` | `u_curr` (Zonal Surface Current) | PO.DAAC OSCAR L4 |
| `4` | `v_curr` (Meridional Surface Current) | PO.DAAC OSCAR L4 |
| `5` | `u_wind` (Zonal Wind at 10m) | PO.DAAC CCMP v3.1 |
| `6` | `v_wind` (Meridional Wind at 10m)| PO.DAAC CCMP v3.1 |

*Note: All features have been Z-score normalized (Mean=0, Std=1). Any missing pixels (e.g., land masks) have been zero-filled to prevent PyTorch NaN propagation.*

---

## 2. Target Data ($Y$ Tensor)
**File**: `data/processed/processed_subsurface_data.nc`
**Shape**: `[Time, 15, Lat, Lon]`

This file contains the historical 3D temperature profiles that the U-Net needs to reconstruct.
| Variable Name | Source Mission / Product |
| :--- | :--- |
| `temperature` | GLORYS12V1 Global Ocean Physics Reanalysis |

The 15 channels correspond linearly to the 15 standard depths specified above. Like the input features, the target temperatures have been Z-score normalized.

---

## 3. PyTorch Dataset Loader

A custom `OceanDataset` class has already been written for you in `src/data/dataset.py`. It inherits from `torch.utils.data.Dataset` and handles loading the NetCDF files into VRAM-efficient PyTorch Tensors.

### Usage Example for ML Team
```python
from torch.utils.data import DataLoader
from src.data.dataset import OceanDataset

# 1. Initialize the dataset
dataset = OceanDataset(processed_dir='data/processed/')

# 2. Get a single day's data (Returns X, Y tensors)
x_sample, y_sample = dataset[0]
print("Input Shape:", x_sample.shape)   # Expected: torch.Size([7, 101, 241])
print("Target Shape:", y_sample.shape)  # Expected: torch.Size([15, 101, 241])

# 3. Pass to PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

for batch_idx, (inputs, targets) in enumerate(dataloader):
    # inputs -> [Batch, 7, 101, 241]
    # targets -> [Batch, 15, 101, 241]
    # Feed to U-Net model here...
    break
```

### De-Normalization
If you need to calculate real-world physical metrics (like RMSE in degrees Celsius), you can reverse the Z-score normalization using the stats saved in `data/processed/normalization_stats.yaml`.

---
*Happy Training!* 🚀
