import os
import pandas as pd
from datetime import datetime

RESULTS_FILE = "results/metrics.csv"


def log_results(model_name, features, test_loss, direction_acc, total_return, sharpe):

    os.makedirs("results", exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "features": features,
        "test_loss": test_loss,
        "direction_accuracy": direction_acc,
        "total_return": total_return,
        "sharpe_ratio": sharpe
    }

    # Case 1: file does NOT exist → create it
    if not os.path.exists(RESULTS_FILE):
        df = pd.DataFrame([row])
        df.to_csv(RESULTS_FILE, index=False)
        return

    # Case 2: file exists but is empty → overwrite it
    if os.path.getsize(RESULTS_FILE) == 0:
        df = pd.DataFrame([row])
        df.to_csv(RESULTS_FILE, index=False)
        return

    # Case 3: normal append
    df = pd.read_csv(RESULTS_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)