import os
import yaml
import glob
import xarray as xr
import numpy as np
import warnings

# Suppress xarray rename coordinate warnings for a clean presentation output
warnings.filterwarnings("ignore", category=UserWarning, message="rename.*does not create an index anymore")

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_variable_name(ds, expected_names):
    for name in expected_names:
        if name in ds.data_vars:
            return name
    return None

def standardize_coords(ds):
    # Fix OSCAR dimensions vs coordinates issue
    if 'latitude' in ds.dims and 'lat' in ds.coords:
        ds = ds.swap_dims({'latitude': 'lat'})
    elif 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
        
    if 'longitude' in ds.dims and 'lon' in ds.coords:
        ds = ds.swap_dims({'longitude': 'lon'})
    elif 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})
    
    rename_dict = {}
    for coord in ['longitude', 'nav_lon', 'lon']:
        if coord in ds.coords or coord in ds.dims: rename_dict[coord] = 'lon'
    for coord in ['latitude', 'nav_lat', 'lat']:
        if coord in ds.coords or coord in ds.dims: rename_dict[coord] = 'lat'
    for coord in ['Time', 't', 'time']:
        if coord in ds.coords or coord in ds.dims: rename_dict[coord] = 'time'
    
    # Remove identical mappings
    rename_dict = {k: v for k, v in rename_dict.items() if k != v}
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    if 'time' in ds.coords and ds['time'].dtype == 'O':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds['time'] = ds.indexes['time'].to_datetimeindex()
            
    return ds

def preprocess_data():
    config = load_config()
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Loading real downloaded datasets...")
    target_lats = np.arange(config['region']['lat_min'], config['region']['lat_max'] + config['region']['resolution'], config['region']['resolution'])
    target_lons = np.arange(config['region']['lon_min'], config['region']['lon_max'] + config['region']['resolution'], config['region']['resolution'])
    
    ds_surface_norm = xr.Dataset(coords={'lat': target_lats, 'lon': target_lons})
    stats = {}
    
    # Group files natively by expected patterns
    groups = {
        'sst': {'glob': 'METOFFICE-GLO-SST*.nc', 'vars': ['sst', 'analysed_sst', 'thetao']},
        'sss': {'glob': 'cmems_obs-mob*.nc', 'vars': ['sss', 'sos', 'so']},
        'ssh': {'glob': 'cmems_obs-sl*.nc', 'vars': ['ssh', 'sla', 'adt', 'zos']},
        'u_curr': {'glob': 'OSCAR*.nc', 'vars': ['u_curr', 'uo', 'u', 'u_current']},
        'v_curr': {'glob': 'OSCAR*.nc', 'vars': ['v_curr', 'vo', 'v', 'v_current']},
        'u_wind': {'glob': 'CCMP*.nc', 'vars': ['u_wind', 'uwnd', 'eastward_wind']},
        'v_wind': {'glob': 'CCMP*.nc', 'vars': ['v_wind', 'vwnd', 'northward_wind']}
    }
    
    for standard_name, info in groups.items():
        files = glob.glob(os.path.join(raw_dir, info['glob']))
        if not files:
            print(f"  -> WARNING: No files found for {standard_name} using glob {info['glob']}")
            continue
            
        print(f"Processing {standard_name} from {len(files)} file(s)...")
        try:
            # Use open_mfdataset for robust multi-granule loading with Dask chunking to prevent OOM
            ds = xr.open_mfdataset(files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override', chunks={'time': 10})
            ds = standardize_coords(ds)
            actual_name = get_variable_name(ds, info['vars'])
            
            if actual_name:
                print(f"  -> Found {standard_name} (as {actual_name})")
                
                # Ensure ascending lat/lon for xarray interpolation
                if ds.lat[0] > ds.lat[-1]:
                    ds = ds.sel(lat=slice(None, None, -1))
                if ds.lon[0] > ds.lon[-1]:
                    ds = ds.sel(lon=slice(None, None, -1))
                
                # Optional: Ensure no duplicate timestamps before interpolating
                _, index = np.unique(ds['time'], return_index=True)
                ds = ds.isel(time=index)
                
                # OPTIMIZATION: Cast to float32 to halve memory usage
                ds[actual_name] = ds[actual_name].astype(np.float32)
                
                ds_regridded = ds.interp(lat=target_lats, lon=target_lons, method='linear')
                arr = ds_regridded[actual_name]
                
                # If there is a 'depth' or 'z' axis equal to 1, squeeze it out
                for dim in ['depth', 'z']:
                    if dim in arr.dims and len(arr[dim]) == 1:
                        arr = arr.squeeze(dim=dim)
                
                mean_val = float(arr.mean().compute().item())
                std_val = float(arr.std().compute().item())
                if std_val == 0 or np.isnan(std_val): std_val = 1e-6
                
                ds_surface_norm[standard_name] = (arr - mean_val) / std_val
                stats[standard_name] = {'mean': mean_val, 'std': std_val}
            else:
                print(f"  -> WARNING: Target variables not found in dataset: {list(ds.data_vars.keys())}")
        except Exception as e:
            print(f"  -> ERROR processing {standard_name}: {e}")

    # Process Target Variable (Subsurface)
    ds_subsurface_norm = xr.Dataset(coords={'lat': target_lats, 'lon': target_lons})
    target_depths = config['depths']['levels']
    glorys_files = glob.glob(os.path.join(raw_dir, 'cmems_mod_glo_phy*.nc'))
    
    if glorys_files:
        try:
            print(f"Processing target subsurface temperature from {len(glorys_files)} file(s)...")
            ds = xr.open_mfdataset(glorys_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override', chunks={'time': 10})
            ds = standardize_coords(ds)
            actual_name = get_variable_name(ds, ['temperature', 'thetao'])
            if actual_name:
                print(f"  -> Found target temperature (as {actual_name})")
                
                if ds.lat[0] > ds.lat[-1]:
                    ds = ds.sel(lat=slice(None, None, -1))
                if ds.lon[0] > ds.lon[-1]:
                    ds = ds.sel(lon=slice(None, None, -1))
                
                _, index = np.unique(ds['time'], return_index=True)
                ds = ds.isel(time=index)
                
                # 1. OPTIMIZATION: Interpolate depth FIRST to reduce memory from 50 levels down to 15 levels!
                if 'depth' in ds.dims:
                    ds = ds.interp(depth=target_depths, method='linear')
                    
                # 2. OPTIMIZATION: Cast to float32 to halve memory usage
                ds[actual_name] = ds[actual_name].astype(np.float32)
                
                ds_regridded = ds.interp(lat=target_lats, lon=target_lons, method='linear')
                arr_target = ds_regridded[actual_name]
                    
                temp_mean = float(arr_target.mean().compute().item())
                temp_std = float(arr_target.std().compute().item())
                if temp_std == 0 or np.isnan(temp_std): temp_std = 1e-6
                
                ds_subsurface_norm['temperature'] = (arr_target - temp_mean) / temp_std
                stats['temperature'] = {'mean': temp_mean, 'std': temp_std}
        except Exception as e:
            print(f"  -> ERROR processing GLORYS: {e}")
    else:
        print("  -> ERROR: GLORYS files not found")

    print("Aligning time axes across all datasets...")
    # Resample all variables to daily start-of-day resolution
    ds_surface_norm = ds_surface_norm.resample(time='1D').mean()
    ds_subsurface_norm = ds_subsurface_norm.resample(time='1D').mean()
    
    # Use pandas index intersection (fixing the previous AttributeError)
    common_time = ds_surface_norm.indexes['time'].intersection(ds_subsurface_norm.indexes['time'])
    print(f"Found {len(common_time)} common daily time steps.")
    
    ds_surface_norm = ds_surface_norm.sel(time=common_time)
    ds_subsurface_norm = ds_subsurface_norm.sel(time=common_time)
    
    with open(os.path.join(processed_dir, 'normalization_stats.yaml'), 'w') as f:
        yaml.dump(stats, f)
        
    ds_surface_norm = ds_surface_norm.fillna(0)
    ds_subsurface_norm = ds_subsurface_norm.fillna(0)
    
    # Ensure consistent dimension ordering before saving
    ds_surface_norm = ds_surface_norm.transpose('time', 'lat', 'lon')
    if 'depth' in ds_subsurface_norm.dims:
        ds_subsurface_norm = ds_subsurface_norm.transpose('time', 'depth', 'lat', 'lon')
    else:
        ds_subsurface_norm = ds_subsurface_norm.transpose('time', 'lat', 'lon')
    
    ds_surface_norm.to_netcdf(os.path.join(processed_dir, "processed_surface_data.nc"))
    ds_subsurface_norm.to_netcdf(os.path.join(processed_dir, "processed_subsurface_data.nc"))
    
    print(f"\nSUCCESS: Real data preprocessed and merged. Saved to {processed_dir}")

if __name__ == "__main__":
    preprocess_data()
