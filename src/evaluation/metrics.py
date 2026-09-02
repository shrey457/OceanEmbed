import torch
import numpy as np

def compute_rmse(pred, target):
    """
    Computes Root Mean Square Error.
    Args:
        pred: torch.Tensor of shape (B, C, H, W)
        target: torch.Tensor of shape (B, C, H, W)
    Returns:
        float: RMSE value
    """
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()

def compute_bias(pred, target):
    """
    Computes Mean Bias (Prediction - Target).
    Args:
        pred: torch.Tensor of shape (B, C, H, W)
        target: torch.Tensor of shape (B, C, H, W)
    Returns:
        float: Bias value
    """
    return torch.mean(pred - target).item()

def compute_correlation(pred, target):
    """
    Computes Pearson Correlation Coefficient across spatial dimensions.
    Args:
        pred: torch.Tensor of shape (B, C, H, W)
        target: torch.Tensor of shape (B, C, H, W)
    Returns:
        float: Mean correlation coefficient
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate means
    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)
    
    # Calculate covariance and variances
    cov = torch.mean((pred_flat - pred_mean) * (target_flat - target_mean))
    pred_var = torch.mean((pred_flat - pred_mean) ** 2)
    target_var = torch.mean((target_flat - target_mean) ** 2)
    
    # Calculate correlation
    corr = cov / (torch.sqrt(pred_var * target_var) + 1e-8)
    return corr.item()

def evaluate_metrics(pred, target):
    """
    Computes all metrics.
    """
    return {
        "RMSE": compute_rmse(pred, target),
        "Bias": compute_bias(pred, target),
        "Correlation": compute_correlation(pred, target)
    }
