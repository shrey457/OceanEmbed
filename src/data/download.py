import os
import yaml
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_synthetic_data():
    """Generates synthetic daily netCDF data for PoC testing."""
    config = load_config()
    raw_dir = config['data']['raw_dir']
    os.makedirs(raw_dir, exist_ok=True)
    
    # Grid definition
    lats = np.arange(config['region']['lat_min'], config['region']['lat_max'] + config['region']['resolution'], config['region']['resolution'])
    lons = np.arange(config['region']['lon_min'], config['region']['lon_max'] + config['region']['resolution'], config['region']['resolution'])
    depths = config['depths']['levels']
    
    # Generate 5 days of data
    start_date = datetime(2023, 1, 1)
    time = [start_date + timedelta(days=i) for i in range(5)]
    
    print(f"Generating synthetic data for {len(time)} days over {len(lats)} lats and {len(lons)} lons...")
    
    # 1. Surface Variables (Inputs)
    # SST, SSS, SSH, U_curr, V_curr, U_wind, V_wind
    surface_vars = ['sst', 'sss', 'ssh', 'u_curr', 'v_curr', 'u_wind', 'v_wind']
    
    ds_surface = xr.Dataset(
        coords={
            'time': time,
            'lat': lats,
            'lon': lons
        }
    )
    
    # Generate some coherent-looking random data (using sine waves + noise)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    base_pattern = np.sin(lon_grid / 10.0) * np.cos(lat_grid / 10.0)
    
    for var in surface_vars:
        # Add random noise for each day
        data = np.zeros((len(time), len(lats), len(lons)))
        for t in range(len(time)):
            data[t, :, :] = base_pattern + np.random.normal(0, 0.1, (len(lats), len(lons)))
            
        ds_surface[var] = (('time', 'lat', 'lon'), data)
        
    surface_path = os.path.join(raw_dir, "synthetic_surface_data.nc")
    ds_surface.to_netcdf(surface_path)
    print(f"Saved surface data to {surface_path}")
    
    # 2. Subsurface Variable (Target - GLORYS Temperature)
    ds_subsurface = xr.Dataset(
        coords={
            'time': time,
            'depth': depths,
            'lat': lats,
            'lon': lons
        }
    )
    
    # Temperature decreases with depth
    temp_data = np.zeros((len(time), len(depths), len(lats), len(lons)))
    for t in range(len(time)):
        for d_idx, depth in enumerate(depths):
            # Base temperature drops exponentially with depth
            base_temp = 30.0 * np.exp(-depth / 300.0) 
            temp_data[t, d_idx, :, :] = base_temp + base_pattern * (1.0 - depth/1000.0) + np.random.normal(0, 0.5, (len(lats), len(lons)))
            
    ds_subsurface['temperature'] = (('time', 'depth', 'lat', 'lon'), temp_data)
    
    subsurface_path = os.path.join(raw_dir, "synthetic_subsurface_data.nc")
    ds_subsurface.to_netcdf(subsurface_path)
    print(f"Saved subsurface data to {subsurface_path}")

if __name__ == "__main__":
    generate_synthetic_data()
