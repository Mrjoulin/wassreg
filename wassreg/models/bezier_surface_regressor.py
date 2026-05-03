import torch
from typing import Optional

from .base import WassersteinSurfaceRegressor
from utils.basis import bernstein_basis


class WassersteinBezierSurfaceRegressor(WassersteinSurfaceRegressor):
    """Bezier surface regressor using PyTorch."""

    def __init__(
        self,
        n_control_u: int = 6,
        n_control_v: int = 6,
        grid_size: int = 50,
        device: Optional[str] = None,
    ) -> None:
        """Initialize Bezier surface regressor.

        Args:
            n_control_u: Control points along u-axis.
            n_control_v: Control points along v-axis.
            grid_size: Grid resolution for variance binning.
            device: Optional device string.
        """
        super().__init__(
            n_control_u=n_control_u,
            n_control_v=n_control_v,
            degree_u=n_control_u - 1,  # Bezier degree = n_cp - 1
            degree_v=n_control_v - 1,
            closed_u=False,
            closed_v=False,
            grid_size=grid_size,
            device=device,
        )

    def _basis_u(self, u_unit: torch.Tensor) -> torch.Tensor:
        """Return Bernstein basis matrix for u direction."""
        return bernstein_basis(u_unit, self.Nu)

    def _basis_v(self, v_unit: torch.Tensor) -> torch.Tensor:
        """Return Bernstein basis matrix for v direction."""
        return bernstein_basis(v_unit, self.Nv)
