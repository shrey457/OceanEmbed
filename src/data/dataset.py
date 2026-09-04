import os
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np

class OceanDataset(Dataset):
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
        
        surface_path = os.path.join(processed_dir, "processed_surface_data.nc")
        subsurface_path = os.path.join(processed_dir, "processed_subsurface_data.nc")
        
        print("Loading datasets into RAM for maximum speed...")
        ds_surface = xr.open_dataset(surface_path).load()
        ds_subsurface = xr.open_dataset(subsurface_path).load()
        
        self.time_len = len(ds_surface['time'])
        self.surface_vars = ['sst', 'sss', 'ssh', 'u_curr', 'v_curr', 'u_wind', 'v_wind']
        
        # 1. Base Satellite Features
        input_data = []
        for var in self.surface_vars:
            input_data.append(ds_surface[var].values)  # Shape: (Time, Lat, Lon)
            
        # 2. Smart Feature: Spatial Awareness (CoordConv)
        # Create normalized Latitude and Longitude grids [-1, 1]
        lats = ds_surface['lat'].values
        lons = ds_surface['lon'].values
        
        lat_norm = 2.0 * (lats - np.min(lats)) / (np.max(lats) - np.min(lats)) - 1.0
        lon_norm = 2.0 * (lons - np.min(lons)) / (np.max(lons) - np.min(lons)) - 1.0
        
        # Expand 1D arrays to full 2D spatial grid, then expand to 3D (Time, Lat, Lon)
        lat_grid, lon_grid = np.meshgrid(lat_norm, lon_norm, indexing='ij')
        lat_feature = np.tile(lat_grid, (self.time_len, 1, 1))
        lon_feature = np.tile(lon_grid, (self.time_len, 1, 1))
        
        input_data.append(lat_feature)
        input_data.append(lon_feature)
        
        # 3. Smart Feature: Temporal Awareness (Seasonality)
        # Convert Day of Year (1-365) to continuous cyclic Sine/Cosine waves
        day_of_year = ds_surface['time'].dt.dayofyear.values
        sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
        cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Expand 1D time arrays to 3D (Time, Lat, Lon)
        _, lat_len, lon_len = lat_feature.shape
        sin_feature = np.broadcast_to(sin_day[:, None, None], (self.time_len, lat_len, lon_len))
        cos_feature = np.broadcast_to(cos_day[:, None, None], (self.time_len, lat_len, lon_len))
        
        input_data.append(sin_feature)
        input_data.append(cos_feature)
        
        # Stack all 11 channels: (11, Time, Lat, Lon) -> Transpose to (Time, 11, Lat, Lon)
        self.input_tensor = torch.tensor(np.stack(input_data), dtype=torch.float32).permute(1, 0, 2, 3)
        self.target_tensor = torch.tensor(ds_subsurface['temperature'].values, dtype=torch.float32)
        
        # Handle land masks (NaN -> 0.0)
        self.input_tensor = torch.nan_to_num(self.input_tensor, nan=0.0)
        self.target_tensor = torch.nan_to_num(self.target_tensor, nan=0.0)
        print("Data successfully loaded and feature engineered in memory!")

    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        # Extremely fast memory slice (no disk I/O)
        return self.input_tensor[idx], self.target_tensor[idx]

def get_dataloaders(config, test_split=0.2):
    processed_dir = config['data']['processed_dir']
    dataset = OceanDataset(processed_dir)
    
    dataset_size = len(dataset)
    test_size = max(1, int(test_split * dataset_size))
    train_size = dataset_size - test_size
    
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, dataset_size))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Adding num_workers and pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader
