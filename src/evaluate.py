import torch


def directional_accuracy(pred, true):
    """Compare whether consecutive predictions move in the same direction as truth."""

    # This version evaluates direction from one step to the next within a series.
    pred_dir = (pred[1:] > pred[:-1])
    true_dir = (true[1:] > true[:-1])
    print("Correlation(pred, true):", np.corrcoef(y_pred.flatten(), y_true.flatten())[0,1])

    return (pred_dir == true_dir).mean()
