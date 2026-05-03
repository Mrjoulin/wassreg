import numpy as np
import matplotlib.pyplot as plt
from wassreg import WassersteinBezierRegressor

# Generate 1D signal: sum of sines + noise
np.random.seed(42)
n_samples = 200
X = np.linspace(0, 1, n_samples).reshape(-1, 1)
y = (
    0.5 * np.sin(2 * np.pi * X) +
    0.3 * np.sin(6 * np.pi * X) +
    0.2 * np.cos(10 * np.pi * X) +
    0.1 * np.random.randn(n_samples, 1)
)

# Fit Bezier curve regressor
model = WassersteinBezierRegressor(n_control_points=10, k_neighbors=15)
model.fit(X, y, epochs=1000, lr=1e-3, verbose=200)

# Predict on test grid
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
mean, std = model.predict(X_test, with_std=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, alpha=0.5, label="Noisy data")
plt.plot(X_test, mean, "r-", linewidth=2, label="Predicted mean")
plt.fill_between(
    X_test.flatten(),
    (mean - std).flatten(),
    (mean + std).flatten(),
    alpha=0.2,
    color="r",
    label="±1 std",
)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Bezier Curve Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
