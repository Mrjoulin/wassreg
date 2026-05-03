# Wasserstein Regressor

PyTorch-based library for Wasserstein-2 regression using Bezier and B-spline curves/surfaces. Fits continuous curves and surfaces with uncertainty estimation via diagonal Wasserstein-2 loss.

## Features

- **1D Curve Fitting**: Bezier and B-spline curves with optional periodicity
- **2D Surface Fitting**: Bezier and B-spline surfaces with per-axis configuration
- **Uncertainty Estimation**: Predicts both mean and variance using Wasserstein-2 loss
- **Local Variance Estimation**: Automatic per-sample variance via sliding window (1D) or grid binning (2D)
- **Device Agnostic**: Auto-detects CUDA/MPS/CPU or manual device selection

## Installation

```bash
# Using uv (recommended)
uv pip install wassreg  # Replace with your chosen name

# Using pip
pip install wassreg
```

For development:
```bash
git clone https://github.com/yourusername/wassreg.git
cd wassreg
uv sync --extra dev
```

## Quick Start

### 1D Curve Fitting (Bezier)

```python
import torch
from WassersteinRegressor import WassersteinBezierRegressor

# Generate sample data
X = torch.linspace(0, 1, 100).reshape(-1, 1)
y = torch.sin(2 * torch.pi * X) + 0.1 * torch.randn(100, 1)

# Fit Bezier curve
model = WassersteinBezierRegressor(n_control_points=7)
model.fit(X, y, epochs=500, lr=1e-3, verbose=50)

# Predict
X_test = torch.linspace(0, 1, 50).reshape(-1, 1)
mean, std = model.predict(X_test, with_std=True)
```

### 1D Curve Fitting (B-Spline)

```python
from WassersteinRegressor import WassersteinBSplineRegressor

# Closed B-spline curve
model = WassersteinBSplineRegressor(
    n_control_points=10,
    degree=3,
    closed=True
)
model.fit(X, y, epochs=500)
```

### 2D Surface Fitting

```python
from WassersteinRegressor import WassersteinBezierSurfaceRegressor
import numpy as np

# Generate 2D data
u = np.random.rand(200)
v = np.random.rand(200)
X = np.column_stack([u, v])
y = np.sin(2 * np.pi * u) * np.cos(2 * np.pi * v) + 0.1 * np.random.randn(200, 1)
y = torch.tensor(y, dtype=torch.float32)

# Fit Bezier surface
model = WassersteinBezierSurfaceRegressor(
    n_control_u=6,
    n_control_v=6,
    grid_size=50
)
model.fit(X, y, epochs=500, lr=1e-3)
```

## API Overview

| Class | Description |
|-------|-------------|
| `WassersteinBezierRegressor` | 1D Bezier curve regression |
| `WassersteinBSplineRegressor` | 1D B-spline curve regression |
| `WassersteinBezierSurfaceRegressor` | 2D Bezier surface regression |
| `WassersteinBSplineSurfaceRegressor` | 2D B-spline surface regression |

## License

MIT
