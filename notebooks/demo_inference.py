import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ocean_embed_cnn import OceanEmbedCNN
from src.data.dataset import OceanDataset

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def visualize_results():
    config = load_config()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")
    
    # Load Model
    model = OceanEmbedCNN(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)
    
    model_path = "models/best_ocean_embed_cnn.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model weights.")
    else:
        print("WARNING: Trained model not found. Using untrained weights for demonstration.")
        
    model.eval()
    
    # Load one sample of data
    try:
        dataset = OceanDataset(config['data']['processed_dir'])
        inputs, targets = dataset[0] # Get the first day
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have run preprocess.py")
        return
        
    inputs = inputs.unsqueeze(0).to(device) # Add batch dimension
    targets = targets.unsqueeze(0).to(device)
    
    # Run Inference
    with torch.no_grad():
        predictions = model(inputs)
        
    # Move to CPU for plotting
    inputs = inputs.squeeze(0).cpu().numpy()
    targets = targets.squeeze(0).cpu().numpy()
    predictions = predictions.squeeze(0).cpu().numpy()
    
    depths = config['depths']['levels']
    surface_vars = ['SST', 'SSS', 'SSH', 'U_curr', 'V_curr', 'U_wind', 'V_wind']
    
    # ---------------------------------------------------------
    # PLOTTING: The "Winning" Visual
    # ---------------------------------------------------------
    print("Generating composite visualization...")
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('OceanEmbed: Subsurface Temperature Reconstruction from Satellite Data', fontsize=20, fontweight='bold', y=0.98)
    
    # Plot 1: The 7 Surface Inputs (Top Row)
    for i in range(7):
        ax = fig.add_subplot(3, 7, i + 1)
        im = ax.imshow(inputs[i], cmap='viridis', origin='lower')
        ax.set_title(f'Input: {surface_vars[i]}')
        ax.axis('off')
        
    # Plot 2: Ground Truth Subsurface Temperatures (Middle Row - showing 7 depths)
    plot_depth_idx = [0, 2, 4, 6, 8, 11, 14] # Indices of depths to plot
    
    for i, d_idx in enumerate(plot_depth_idx):
        ax = fig.add_subplot(3, 7, 7 + i + 1)
        im = ax.imshow(targets[d_idx], cmap='coolwarm', origin='lower')
        ax.set_title(f'True Temp @ {depths[d_idx]}m')
        ax.axis('off')
        
    # Plot 3: Predicted Subsurface Temperatures (Bottom Row)
    for i, d_idx in enumerate(plot_depth_idx):
        ax = fig.add_subplot(3, 7, 14 + i + 1)
        im = ax.imshow(predictions[d_idx], cmap='coolwarm', origin='lower')
        ax.set_title(f'Pred Temp @ {depths[d_idx]}m')
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plot_path = "results/ocean_embed_inference.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"SUCCESS: Visualization saved to {plot_path}")
    print("Show this image to the judges!")

if __name__ == "__main__":
    visualize_results()
