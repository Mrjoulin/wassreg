import numpy as np
import matplotlib.pyplot as plt
from wassreg import WassersteinBSplineRegressor

# Generate 1D signal: sum of sines + noise
np.random.seed(42)
n_samples = 1000
a, b, delta = 5, 4, np.pi / 2

X = np.linspace(0, 1, n_samples)
y = np.column_stack([
    np.sin(a * X * 2 * np.pi + delta),
    np.sin(b * X * 2 * np.pi),
])
y += np.random.normal(0, 0.1, size=(n_samples, 2))

# Fit closed B-spline curve regressor
model = WassersteinBSplineRegressor(n_control_points=32, degree=3, closed=True, k_neighbors=15)
model.fit(X, y, epochs=1000, lr=1e-2, verbose=200)

# Predict on test grid
X_test = np.linspace(0, 1, 200).reshape(-1, 1)
mean, std = model.predict(X_test, with_std=True)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(y[:, 0], y[:, 1], s=15, alpha=0.4, label="Noisy data")
plt.plot(mean[:, 0], mean[:, 1], "r-", linewidth=2, label="Predicted mean")
plt.fill_between(
    mean[:, 0],
    (mean[:, 1] - std[:, 1]),
    (mean[:, 1] + std[:, 1]),
    alpha=0.2,
    color="r",
    label="Y ±1 std",
)
plt.fill_between(
    mean[:, 1],
    (mean[:, 0] - std[:, 0]),
    (mean[:, 0] + std[:, 0]),
    alpha=0.2,
    color="g",
    label="X ±1 std",
)
plt.xlabel("X")
plt.ylabel("y")
plt.title("B-Spline Curve Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
