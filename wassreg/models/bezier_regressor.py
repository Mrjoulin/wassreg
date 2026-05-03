import torch
from typing import Optional

from .base import WassersteinCurveRegressor


class WassersteinBezierRegressor(WassersteinCurveRegressor):
    """Bezier curve regressor using De Casteljau algorithm."""

    def __init__(self, n_control_points: int = 7, k_neighbors: int = 10, device: Optional[str] = None):
        """Initialize Bezier curve regressor.

        Args:
            n_control_points: Number of control points (degree = n_cp -1).
            k_neighbors: Window size for local variance estimation.
            device: Optional device string.
        """
        super().__init__(
            n_control_points=n_control_points,
            degree=n_control_points - 1,  # Bezier degree = n_cp - 1
            closed=False,
            k_neighbors=k_neighbors,
            device=device,
        )
        self.N = n_control_points - 1

    def _basis_matrix(self, x_unit: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Basis matrix calculation not needed for this curve")

    def _evaluate(self, t: torch.Tensor) -> tuple:
        """Evaluate Bezier curve using De Casteljau algorithm."""
        t = t.view(-1, 1, 1)
        mu = self.mu_P.unsqueeze(0).repeat(t.shape[0], 1, 1)
        logvar = self.logvar_P.unsqueeze(0).repeat(t.shape[0], 1, 1)
        for _ in range(self.N):
            mu = (1 - t) * mu[:, :-1] + t * mu[:, 1:]
            logvar = (1 - t) * logvar[:, :-1] + t * logvar[:, 1:]
        mu_t = mu.squeeze(1)
        sigma_t = torch.exp(logvar.squeeze(1))
        return mu_t, sigma_t
