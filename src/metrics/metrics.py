import torch
import numpy as np

def directional_accuracy(pred, true):
    if isinstance(pred, np.ndarray):
        pred_dir = np.sign(pred)
        true_dir = np.sign(true)
        return (pred_dir == true_dir).mean()

    pred_dir = torch.sign(pred)
    true_dir = torch.sign(true)
    return (pred_dir == true_dir).float().mean().item()