import os
import yaml
import datetime
from dotenv import load_dotenv

try:
    import copernicusmarine
except ImportError:
    print("Please install copernicusmarine: pip install copernicusmarine")
    
try:
    import earthaccess
except ImportError:
    print("Please install earthaccess: pip install earthaccess")

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_copernicus_dataset(dataset_id, output_dir, config, start_date, end_date):
    """
    Downloads a subset of a Copernicus Marine dataset based on spatial and temporal bounds.
    Requires COPERNICUS_MARINE_SERVICE_USERNAME and COPERNICUS_MARINE_SERVICE_PASSWORD in environment.
    """
    print(f"Downloading Copernicus dataset: {dataset_id}")
    
    output_filename = f"{dataset_id}.nc"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        print(f"--> File {output_filename} already exists. Skipping download.")
        return
        
    # We use copernicusmarine.subset to only download our bounding box and time range
    copernicusmarine.subset(
        dataset_id=dataset_id,
        output_directory=output_dir,
        output_filename=output_filename,
        start_datetime=start_date,
        end_datetime=end_date,
        minimum_longitude=config['region']['lon_min'],
        maximum_longitude=config['region']['lon_max'],
        minimum_latitude=config['region']['lat_min'],
        maximum_latitude=config['region']['lat_max'],
        overwrite=False
    )

def download_podaac_dataset(short_name, output_dir, config, start_date, end_date):
    """
    Downloads NASA PO.DAAC data using earthaccess.
    Requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD in environment.
    """
    print(f"Downloading NASA Earthdata dataset: {short_name}")
    earthaccess.login(strategy="environment") # Assumes env vars are set
    
    # Search for granules
    bounding_box = (
        config['region']['lon_min'], 
        config['region']['lat_min'], 
        config['region']['lon_max'], 
        config['region']['lat_max']
    )
    
    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bounding_box,
        temporal=(start_date, end_date)
    )
    
    if not results:
        print(f"No granules found for {short_name} in the given space/time bounds.")
        return
        
    print(f"Found {len(results)} granules. Downloading...")
    earthaccess.download(results, output_dir)

def main():
    load_dotenv() # Load variables from .env if present
    config = load_config()
    raw_dir = config['data']['raw_dir']
    os.makedirs(raw_dir, exist_ok=True)
    
    # Define time range for downloading (1 full year for robust temporal validation)
    start_date = "2023-01-01 00:00:00"
    end_date = "2023-12-31 23:59:59"
    podaac_start = "2023-01-01"
    podaac_end = "2023-12-31"
    
    print(f"Starting actual data download for period: {start_date} to {end_date}")
    print("Ensure you have set COPERNICUS_MARINE_SERVICE_USERNAME, COPERNICUS_MARINE_SERVICE_PASSWORD,")
    print("EARTHDATA_USERNAME, and EARTHDATA_PASSWORD in your environment or .env file.\n")
    
    # --- Copernicus Marine Datasets ---
    # 1. Target Subsurface Temperature: GLORYS Reanalysis (moi-00021)
    download_copernicus_dataset("cmems_mod_glo_phy_my_0.083deg_P1D-m", raw_dir, config, start_date, end_date)
    
    # 2. SST: OSTIA (moi-00168)
    download_copernicus_dataset("METOFFICE-GLO-SST-L4-REP-OBS-SST", raw_dir, config, start_date, end_date)
    
    # 3. SSS: SMOS/SMAP (moi-00051)
    download_copernicus_dataset("cmems_obs-mob_glo_phy-sss_my_multi_P1D", raw_dir, config, start_date, end_date)
    
    # 4. SSH: DUACS (moi-00148)
    download_copernicus_dataset("cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", raw_dir, config, start_date, end_date)
    
    # --- NASA PO.DAAC Datasets ---
    # 5. Surface Currents: OSCAR_L4OC_FINAL_V2.0
    download_podaac_dataset("OSCAR_L4_OC_FINAL_V2.0", raw_dir, config, podaac_start, podaac_end)
    
    # 6. Surface Winds: CCMP_WINDS_10M6HR_L4_V3.1
    download_podaac_dataset("CCMP_WINDS_10M6HR_L4_V3.1", raw_dir, config, podaac_start, podaac_end)
    
    print("Download complete. You will need to update src/data/preprocess.py to handle the specific variable names in these actual NetCDF files.")

if __name__ == "__main__":
    main()
