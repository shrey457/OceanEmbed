from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import yaml
import os
import sys
import numpy as np

# Add src to path so we can import model and dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ocean_embed_cnn import OceanEmbedCNN
from src.data.dataset import OceanDataset

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_config():
    with open("config.yaml", 'r') as file:
        return yaml.safe_load(file)

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for caching
model = None
dataset = None
temp_mean = 0.0
temp_std = 1.0

@app.on_event("startup")
def startup_event():
    global model, dataset
    
    # Load Model
    model = OceanEmbedCNN(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)
    
    model_path = "models/best_ocean_embed_cnn.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model weights.")
    model.eval()
    
    # Load Dataset
    dataset = OceanDataset(config['data']['processed_dir'])
    print(f"Dataset loaded with {len(dataset)} days of data.")
    
    # Load Normalization Stats
    stats_path = os.path.join(config['data']['processed_dir'], "normalization_stats.yaml")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)
            if 'temperature' in stats:
                global temp_mean, temp_std
                temp_mean = stats['temperature']['mean']
                temp_std = stats['temperature']['std']
                print(f"Loaded temperature denormalization stats: mean={temp_mean:.2f}, std={temp_std:.2f}")

@app.get("/api/config")
def get_config():
    # Return basic config for the frontend (depth levels, lats, lons)
    lats = np.arange(config['region']['lat_min'], config['region']['lat_max'] + config['region']['resolution'], config['region']['resolution']).tolist()
    lons = np.arange(config['region']['lon_min'], config['region']['lon_max'] + config['region']['resolution'], config['region']['resolution']).tolist()
    return {
        "depths": config['depths']['levels'],
        "lats": lats,
        "lons": lons,
        "num_days": len(dataset)
    }

@app.get("/api/predict/{day_idx}")
def predict(day_idx: int):
    global model, dataset
    if day_idx < 0 or day_idx >= len(dataset):
        return {"error": "Invalid day index"}
        
    inputs, targets = dataset[day_idx]
    
    inputs_tensor = inputs.unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(inputs_tensor).squeeze(0).cpu().numpy()
        
    targets = targets.cpu().numpy()
    
    # Find land mask based on the exact 0.0 padding in normalized targets
    land_mask = (targets == 0.0)
    
    # Denormalize to Celsius
    targets = (targets * temp_std) + temp_mean
    predictions = (predictions * temp_std) + temp_mean
    
    # Apply land mask
    targets[land_mask] = np.nan
    predictions[land_mask] = np.nan
    
    # Convert nan to None for JSON Plotly rendering
    def clean_array(arr):
        return np.where(np.isnan(arr), None, arr).tolist()
        
    return {
        "true": clean_array(targets),
        "pred": clean_array(predictions)
    }

# Serve static files
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("src/app/static/index.html")
