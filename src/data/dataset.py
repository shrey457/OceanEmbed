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
        
        self.ds_surface = xr.open_dataset(surface_path)
        self.ds_subsurface = xr.open_dataset(subsurface_path)
        
        self.time_len = len(self.ds_surface['time'])
        self.surface_vars = ['sst', 'sss', 'ssh', 'u_curr', 'v_curr', 'u_wind', 'v_wind']

    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        # Input shape: (Channels, Height, Width) -> (7, Lat, Lon)
        input_data = []
        for var in self.surface_vars:
            # Extract spatial grid for given day
            val = self.ds_surface[var].isel(time=idx).values
            input_data.append(val)
        
        input_tensor = torch.tensor(np.stack(input_data), dtype=torch.float32)
        
        # Target shape: (Channels, Height, Width) -> (15 depths, Lat, Lon)
        target_val = self.ds_subsurface['temperature'].isel(time=idx).values
        target_tensor = torch.tensor(target_val, dtype=torch.float32)
        
        # In real data, land masks appear as NaNs. We replace them with 0 for the model.
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)
        
        return input_tensor, target_tensor

def get_dataloaders(config, test_split=0.2):
    processed_dir = config['data']['processed_dir']
    dataset = OceanDataset(processed_dir)
    
    dataset_size = len(dataset)
    test_size = max(1, int(test_split * dataset_size))
    train_size = dataset_size - test_size
    
    # We use strict chronological splitting to prevent temporal data leakage.
    # The model trains on the past and validates on the unseen future.
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, dataset_size))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, test_loader
