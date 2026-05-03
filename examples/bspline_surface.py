import numpy as np
import plotly.graph_objects as go
from wassreg import WassersteinBSplineSurfaceRegressor


# Generate 3D Tor
n_samples = 2000
R, r = 10, 4

u = np.random.rand(n_samples)
v = np.random.rand(n_samples)
X = np.column_stack([u, v])

x = (R + r * np.cos(u * 2 * np.pi)) * np.cos(v * 2 * np.pi)
y = (R + r * np.cos(u * 2 * np.pi)) * np.sin(v * 2 * np.pi)
z = r * np.sin(u * 2 * np.pi)

y = np.column_stack([x, y, z])
y += np.random.normal(0, 0.1, size=(n_samples, 3))

# Fit B-spline surface regressor
model = WassersteinBSplineSurfaceRegressor(
    n_control_u=8,
    n_control_v=8,
    degree_u=3,
    degree_v=3,
    closed_u=True,
    closed_v=True,
    grid_size=50,
)
model.fit(X, y, epochs=50, lr=0.1, verbose=10, random_batch=False)

# Predict on regular grid
grid_size = 100
u_grid = np.linspace(0, 1, grid_size)
v_grid = np.linspace(0, 1, grid_size)
U, V = np.meshgrid(u_grid, v_grid)
X_grid = np.stack([U.ravel(), V.ravel()], axis=1)

mean, std = model.predict(X_grid, with_std=True)

# Reshape for plotting
Z_mean = np.moveaxis(mean, 1, 0).reshape(3, grid_size, grid_size)

# Plot with plotly
fig = go.Figure(data=[
    go.Scatter3d(
        x=y[:, 0], y=y[:, 1], z=y[:, 2],
        mode='markers', marker=dict(size=1, color='blue', colorscale='Viridis')
    ),
    go.Surface(x=Z_mean[0], y=Z_mean[1], z=Z_mean[2], colorscale="Viridis", name="Mean")
])
fig.update_layout(
    title="B-Spline Surface Regression",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    width=900,
    height=700,
)
fig.show()
