import torch


def directional_accuracy(pred, target, last_price):

    pred_move = pred - last_price
    true_move = target - last_price

    pred_dir = torch.sign(pred_move)
    true_dir = torch.sign(true_move)

    correct = (pred_dir == true_dir).float()

    return correct.mean().item()