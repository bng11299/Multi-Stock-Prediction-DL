import torch


def directional_accuracy(pred, true):

    pred_dir = (pred[1:] > pred[:-1])
    true_dir = (true[1:] > true[:-1])

    return (pred_dir == true_dir).mean()