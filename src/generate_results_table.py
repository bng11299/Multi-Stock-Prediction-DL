import pandas as pd
from pathlib import Path

RESULTS_FILE = Path("results/metrics.csv")

df = pd.read_csv(RESULTS_FILE)

# Standardize column naming
if "direction_acc" in df.columns:
    df["directional_accuracy"] = df["direction_acc"]

columns = [
    "timestamp",
    "model",
    "features",
    "test_loss",
    "directional_accuracy",
    "total_return",
    "sharpe_ratio"
]

# Keep only available columns
columns = [col for col in columns if col in df.columns]

results_table = df[columns]

# Save CSV
results_table.to_csv("results/final_results_table.csv", index=False)

# Save readable TXT table
with open("results/final_results_table.txt", "w") as f:
    f.write(results_table.to_string(index=False))

print("Final results table generated:")
print(results_table)