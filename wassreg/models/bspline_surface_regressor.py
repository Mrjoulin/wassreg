import torch
from typing import Optional

from .base import WassersteinSurfaceRegressor
from wassreg.utils.basis import bspline_basis


class WassersteinBSplineSurfaceRegressor(WassersteinSurfaceRegressor):
    """B-spline surface regressor using PyTorch."""

    def __init__(
        self,
        n_control_u: int = 6,
        n_control_v: int = 6,
        degree_u: int = 3,
        degree_v: int = 3,
        closed_u: bool = False,
        closed_v: bool = False,
        grid_size: int = 50,
        device: Optional[str] = None,
    ) -> None:
        """Initialize B-spline surface regressor.

        Args:
            n_control_u: Control points along u-axis.
            n_control_v: Control points along v-axis.
            degree_u: Spline degree for u-axis.
            degree_v: Spline degree for v-axis.
            closed_u: Whether u-axis is closed/periodic.
            closed_v: Whether v-axis is closed/periodic.
            grid_size: Grid resolution for variance binning.
            device: Optional device string.
        """
        super().__init__(
            n_control_u=n_control_u,
            n_control_v=n_control_v,
            degree_u=degree_u,
            degree_v=degree_v,
            closed_u=closed_u,
            closed_v=closed_v,
            grid_size=grid_size,
            device=device,
        )

    def _basis_u(self, u_unit: torch.Tensor) -> torch.Tensor:
        """Return B-spline basis matrix for u direction."""
        return bspline_basis(u_unit, self.Nu + 1, self.degree_u, closed=self.closed_u)

    def _basis_v(self, v_unit: torch.Tensor) -> torch.Tensor:
        """Return B-spline basis matrix for v direction."""
        return bspline_basis(v_unit, self.Nv + 1, self.degree_v, closed=self.closed_v)
