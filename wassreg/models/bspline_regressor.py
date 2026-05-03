import torch
from typing import Optional

from .base import WassersteinCurveRegressor
from utils.basis import bspline_basis


class WassersteinBSplineRegressor(WassersteinCurveRegressor):
    """B-spline curve regressor implemented with PyTorch."""

    def __init__(
        self,
        n_control_points: int,
        degree: int = 3,
        closed: bool = False,
        k_neighbors: int = 10,
        device: Optional[str] = None,
    ) -> None:
        """Initialize B-spline curve regressor.

        Args:
            n_control_points: Number of control points.
            degree: B-spline degree.
            closed: Whether curve is closed/periodic.
            k_neighbors: Window size for local variance estimation.
            device: Optional device string.
        """
        super().__init__(
            n_control_points=n_control_points,
            degree=degree,
            closed=closed,
            k_neighbors=k_neighbors,
            device=device,
        )

    def _basis_matrix(self, x_unit: torch.Tensor) -> torch.Tensor:
        """Return B-spline basis matrix for normalized parameters."""
        return bspline_basis(x_unit, self.n_control_points, self.degree, closed=self.closed)
