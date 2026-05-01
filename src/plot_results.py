import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = Path("results/metrics.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Load results
# ---------------------------
df = pd.read_csv(RESULTS_FILE)

# Remove malformed rows
df = df.dropna(subset=["model", "features", "direction_accuracy", "test_loss"])

# Optional: convert timestamp if valid
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Create experiment label
df["label"] = df["model"] + " | " + df["features"]

# ---------------------------
# Directional Accuracy Plot
# ---------------------------
plt.figure(figsize=(12, 6))

for label in df["label"].unique():
    subset = df[df["label"] == label]
    plt.plot(
        subset.index,
        subset["direction_accuracy"],
        marker="o",
        label=label
    )

plt.title("Directional Accuracy Across Experiments")
plt.xlabel("Experiment Run")
plt.ylabel("Directional Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "direction_accuracy.png")
plt.close()

# ---------------------------
# Test Loss Plot
# ---------------------------
plt.figure(figsize=(12, 6))

for label in df["label"].unique():
    subset = df[df["label"] == label]
    plt.plot(
        subset.index,
        subset["test_loss"],
        marker="o",
        label=label
    )

plt.title("Test Loss Across Experiments")
plt.xlabel("Experiment Run")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_loss.png")
plt.close()

# ---------------------------
# Sharpe Ratio Plot
# ---------------------------
if "sharpe_ratio" in df.columns:
    plt.figure(figsize=(12, 6))

    for label in df["label"].unique():
        subset = df[df["label"] == label]
        plt.plot(
            subset.index,
            subset["sharpe_ratio"],
            marker="o",
            label=label
        )

    plt.title("Sharpe Ratio Across Experiments")
    plt.xlabel("Experiment Run")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sharpe_ratio.png")
    plt.close()

print("Saved:")
print("- directional_accuracy.png")
print("- test_loss.png")
print("- sharpe_ratio.png")