import torch
import numpy as np

def directional_accuracy(pred, true):
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)
    if isinstance(true, np.ndarray):
        true = torch.tensor(true)

    pred_dir = torch.sign(pred)
    true_dir = torch.sign(true)

    return (pred_dir == true_dir).float().mean().item()