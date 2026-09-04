import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_dataloaders
from models.ocean_embed_cnn import OceanEmbedCNN
from evaluation.metrics import evaluate_metrics

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train():
    config = load_config()
    
    # 1. Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Setup DataLoaders
    try:
        train_loader, test_loader = get_dataloaders(config)
    except FileNotFoundError:
        print("Processed data not found. Please run download.py and then preprocess.py.")
        return
        
    # 3. Initialize Model
    model = OceanEmbedCNN(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)
    
    # 4. Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # 5. Training Loop
    epochs = config['training']['epochs']
    best_loss = float('inf')
    
    # 5.1 Optimization: PyTorch 2.0 Compile
    if hasattr(torch, "compile") and sys.platform != "win32":
        # Note: torch.compile isn't fully supported on Windows yet, but we will add the check
        pass
    
    # 5.2 Optimization: Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # We use a progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Use AMP autocast
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_metrics = {"RMSE": 0, "Bias": 0, "Correlation": 0}
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Validation also benefits from AMP for speed
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item()
                
                # Compute metrics
                batch_metrics = evaluate_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v
                    
        avg_val_loss = val_loss / len(test_loader)
        
        for k in val_metrics.keys():
            val_metrics[k] /= len(test_loader)
            
        print(f"Epoch {epoch+1} Summary: "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"RMSE: {val_metrics['RMSE']:.4f} | "
              f"Bias: {val_metrics['Bias']:.4f} | "
              f"Corr: {val_metrics['Correlation']:.4f}")
              
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_ocean_embed_cnn.pth")
            print("Saved new best model.")
            
    print("Training complete.")

if __name__ == "__main__":
    train()
