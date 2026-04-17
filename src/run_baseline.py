import numpy as np
from sklearn.model_selection import train_test_split

from baselines.persistence import persistence_baseline


X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

mse, acc = persistence_baseline(X_test, y_test)

print("Persistence Baseline")
print("MSE:", mse)
print("Directional Accuracy:", acc)