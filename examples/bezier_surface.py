import numpy as np
import plotly.graph_objects as go
from wassreg import WassersteinBezierSurfaceRegressor

# Generate 2D surface with noise
n_samples = 500
u = np.random.rand(n_samples)
v = np.random.rand(n_samples)
X = np.stack([u, v], axis=1)

r = np.sqrt((u-0.5)**2 + (v-0.5)**2)
y = np.exp(-20*r**2)
y += 0.1 * np.random.randn(n_samples)

# Fit Bezier surface regressor
model = WassersteinBezierSurfaceRegressor(
    n_control_u=8,
    n_control_v=8,
    grid_size=60,
)
model.fit(X, y, epochs=1500, lr=1e-3, verbose=300)

# Predict on regular grid
grid_size = 100
u_grid = np.linspace(0, 1, grid_size)
v_grid = np.linspace(0, 1, grid_size)
U, V = np.meshgrid(u_grid, v_grid)
X_grid = np.stack([U.ravel(), V.ravel()], axis=1)

mean, std = model.predict(X_grid, with_std=True)

# Reshape for plotting
Z_mean = mean.reshape(grid_size, grid_size)
Z_std = std.reshape(grid_size, grid_size)

# Plot with plotly
fig = go.Figure(data=[
    go.Scatter3d(x=u, y=v, z=y, mode='markers', marker=dict(size=1, color='blue', colorscale='Viridis')),
    go.Surface(x=u_grid, y=v_grid, z=Z_mean, colorscale="Viridis", name="Mean",),
    go.Surface(x=u_grid, y=v_grid, z=Z_mean + Z_std, opacity=0.3, colorscale="Reds", name="+1 std"),
    go.Surface(x=u_grid, y=v_grid, z=Z_mean - Z_std, opacity=0.3, colorscale="Blues", name="-1 std"),
])
fig.update_layout(
    title="Bezier Surface Regression",
    scene=dict(xaxis_title="u", yaxis_title="v", zaxis_title="y"),
    width=900,
    height=700,
)
fig.show()
