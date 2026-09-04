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
    
    # Load normalization stats to denormalize targets and predictions
    stats_path = os.path.join(config['data']['processed_dir'], "normalization_stats.yaml")
    temp_mean, temp_std = 0.0, 1.0
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = yaml.safe_load(f)
            if 'temperature' in stats:
                temp_mean = stats['temperature']['mean']
                temp_std = stats['temperature']['std']
                
    # Denormalize targets and predictions
    # Before denormalizing, extract the land mask (0.0 in normalized space)
    land_mask = (targets == 0.0)
    
    targets = (targets * temp_std) + temp_mean
    predictions = (predictions * temp_std) + temp_mean
    
    # Apply land mask (set to NaN so it plots white/transparent instead of a solid color)
    targets[land_mask] = np.nan
    predictions[land_mask] = np.nan
    
    depths = config['depths']['levels']
    surface_vars = ['SST', 'SSS', 'SSH', 'U_curr', 'V_curr', 'U_wind', 'V_wind']
    
    # ---------------------------------------------------------
    # PLOTTING: Enhanced "Winning" Visual
    # ---------------------------------------------------------
    print("Generating comprehensive visualization...")
    # Increase figure size to fit 4 rows + scatter + depth profile
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('OceanEmbed: Subsurface Temperature Reconstruction Analysis', fontsize=24, fontweight='bold', y=0.98)
    
    # Grid spec for better layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, 7, height_ratios=[1, 1, 1, 1, 1.5], hspace=0.3, wspace=0.1)
    
    plot_depth_idx = [0, 2, 4, 6, 8, 11, 14] # Indices of depths to plot
    
    # 1. Plot 7 Surface Inputs (Row 0)
    for i in range(7):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(inputs[i], cmap='viridis', origin='lower')
        ax.set_title(f'Input: {surface_vars[i]}', fontsize=12)
        ax.axis('off')
        
    # 2 & 3 & 4. True, Predicted, and Error (Rows 1, 2, 3)
    for i, d_idx in enumerate(plot_depth_idx):
        # FIX: Calculate consistent color limits for True and Predicted so colors match perfectly!
        vmin = min(targets[d_idx].min(), predictions[d_idx].min())
        vmax = max(targets[d_idx].max(), predictions[d_idx].max())
        
        # True
        ax_true = fig.add_subplot(gs[1, i])
        ax_true.imshow(targets[d_idx], cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        ax_true.set_title(f'Ground Truth @ {depths[d_idx]}m', fontsize=12)
        ax_true.axis('off')
        
        # Predicted
        ax_pred = fig.add_subplot(gs[2, i])
        ax_pred.imshow(predictions[d_idx], cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f'Pred @ {depths[d_idx]}m', fontsize=12)
        ax_pred.axis('off')
        
        # Error (Predicted - True)
        error = predictions[d_idx] - targets[d_idx]
        ax_err = fig.add_subplot(gs[3, i])
        # Center error colormap around 0 (White)
        err_max = max(abs(error.min()), abs(error.max()))
        ax_err.imshow(error, cmap='seismic', origin='lower', vmin=-err_max, vmax=err_max)
        ax_err.set_title(f'Error Map', fontsize=12)
        ax_err.axis('off')

    # 5. Global Scatter Plot (Row 4, spanning columns 0 to 2)
    ax_scatter = fig.add_subplot(gs[4, 0:3])
    # Flatten arrays and mask out nans (land)
    flat_targets = targets.flatten()
    flat_preds = predictions.flatten()
    mask = ~np.isnan(flat_targets)
    flat_targets = flat_targets[mask]
    flat_preds = flat_preds[mask]
    
    # Subsample for scatter plot if too large
    if len(flat_targets) > 10000:
        idx = np.random.choice(len(flat_targets), 10000, replace=False)
        flat_targets_sub = flat_targets[idx]
        flat_preds_sub = flat_preds[idx]
    else:
        flat_targets_sub = flat_targets
        flat_preds_sub = flat_preds
        
    ax_scatter.scatter(flat_targets_sub, flat_preds_sub, alpha=0.1, color='blue', s=2)
    
    # Perfect 1:1 line
    if len(flat_targets) > 0:
        min_val = min(flat_targets.min(), flat_preds.min())
        max_val = max(flat_targets.max(), flat_preds.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect 1:1 Match')
        
    ax_scatter.set_title('Global Scatter Plot: True vs Predicted', fontsize=14)
    ax_scatter.set_xlabel('True Temperature (°C)', fontsize=12)
    ax_scatter.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)

    # 6. Depth Profile Line Graph (Row 4, spanning columns 4 to 6)
    ax_profile = fig.add_subplot(gs[4, 4:7])
    # Calculate spatial mean for each depth level (excluding land)
    true_mean_profile = []
    pred_mean_profile = []
    for d in range(len(depths)):
        t_slice = targets[d]
        p_slice = predictions[d]
        mask_d = ~np.isnan(t_slice)
        true_mean_profile.append(t_slice[mask_d].mean() if mask_d.sum() > 0 else 0)
        pred_mean_profile.append(p_slice[mask_d].mean() if mask_d.sum() > 0 else 0)
        
    ax_profile.plot(true_mean_profile, depths, 'b-o', label='True Average Profile', linewidth=2)
    ax_profile.plot(pred_mean_profile, depths, 'r-*', label='Predicted Average Profile', linewidth=2)
    ax_profile.invert_yaxis() # Depth goes down (0 at top, 1000 at bottom)
    ax_profile.set_title('Average Temperature Profile over Depth', fontsize=14)
    ax_profile.set_xlabel('Temperature (°C)', fontsize=12)
    ax_profile.set_ylabel('Depth (meters)', fontsize=12)
    ax_profile.legend()
    ax_profile.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plot_path = "results/ocean_embed_inference.png"
    # Use facecolor white to ensure transparency issues don't happen
    fig.patch.set_facecolor('white')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"SUCCESS: Visualization saved to {plot_path}")
    print("Show this image to the judges!")

if __name__ == "__main__":
    visualize_results()
