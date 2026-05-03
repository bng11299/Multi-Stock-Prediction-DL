import pandas as pd
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

HISTORY_FILE = RESULTS_DIR / "training_history.csv"


def log_epoch(model_name, features, epoch, train_loss, test_loss, direction_acc):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "features": features,
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "directional_accuracy": direction_acc,
    }

    if HISTORY_FILE.exists():
        try:
            df = pd.read_csv(HISTORY_FILE)
        except:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)