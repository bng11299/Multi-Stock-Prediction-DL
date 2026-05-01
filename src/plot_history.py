import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = PROJECT_ROOT / "results" / "history"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

history_files = list(HISTORY_DIR.glob("*.csv"))

if not history_files:
    print(f"No history CSV files found in {HISTORY_DIR}")

for file in history_files:
    df = pd.read_csv(file)

    name = file.stem

    # Loss plot
    plt.figure(figsize=(10,5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss")
    plt.title(f"{name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_loss.png")
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10,5))
    plt.plot(df["epoch"], df["direction_acc"], label="Directional Accuracy")
    plt.title(f"{name} Directional Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_accuracy.png")
    plt.close()

print("History plots generated.")
