from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

from utils.utils import get_device, w2_diag_loss


class WassersteinRegressor(nn.Module, ABC):
    """Base class with common utilities for Wasserstein regression models."""

    def __init__(
        self,
        k_neighbors: int = 10,
        grid_size: int = 50,
        device: Optional[str] = None,
    ) -> None:
        """Initialize base Wasserstein regressor.

        Args:
            k_neighbors: Window size for local variance estimation.
            grid_size: Grid resolution for 2D surface binning.
            device: Optional device string; auto-detects if None.
        """
        super().__init__()
        self.k = int(k_neighbors)
        self.grid_size = int(grid_size)
        self.device = get_device(device)
        self._normalize_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.mu_P: Optional[nn.Parameter] = None
        self.logvar_P: Optional[nn.Parameter] = None

    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Scale ``X`` to the unit interval and cache min/max for later calls."""
        if self._normalize_params is None:
            self._normalize_params = (X.min(), X.max())
        X_min, X_max = self._normalize_params
        X_norm = (X - X_min) / (X_max - X_min + 1e-10)
        return torch.clamp(X_norm, 0.0, 1.0)

    @abstractmethod
    def estimate_local_variance(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimate per-sample variance."""
        raise NotImplementedError()

    @abstractmethod
    def _initialize_control_points(
        self, X_norm: torch.Tensor, y: torch.Tensor, sigma_y: torch.Tensor
    ) -> None:
        """Create ``mu_P`` and ``logvar_P`` parameters from data."""
        raise NotImplementedError()

    @abstractmethod
    def _evaluate(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and variance predictions for given inputs."""
        raise NotImplementedError()

    def _fit_epoch_random_batch(
        self,
        X_norm: torch.Tensor,
        y: torch.Tensor,
        sigma_y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
    ) -> float:
        """Select random batch and make one train step"""
        batch_idx = torch.randperm(y.shape[0])[:batch_size]
        X_batch = X_norm[batch_idx]
        y_batch = y[batch_idx]
        sigma_y_batch = sigma_y[batch_idx]

        optimizer.zero_grad()
        mu_pred, sigma_pred = self._evaluate(X_batch)
        loss = w2_diag_loss(mu_pred, sigma_pred, y_batch, sigma_y_batch)
        loss.backward()
        optimizer.step()

        return loss.item()

    def _fit_epoch_full(
        self,
        X_norm: torch.Tensor,
        y: torch.Tensor,
        sigma_y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
    ) -> float:
        """Shuffle data and train model on all batches"""
        n = y.shape[0]
        n_batches = int(np.ceil(n / batch_size))
        epoch_idx = torch.randperm(n)
        total_loss = 0
        for ind in range(n_batches):
            batch_idx = epoch_idx[ind * batch_size:(ind + 1) * batch_size]
            X_batch = X_norm[batch_idx]
            y_batch = y[batch_idx]
            sigma_y_batch = sigma_y[batch_idx]

            optimizer.zero_grad()
            mu_pred, sigma_pred = self._evaluate(X_batch)
            loss = w2_diag_loss(mu_pred, sigma_pred, y_batch, sigma_y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / n_batches

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        const_variance: Optional[float] = None,
        batch_size: int = 256,
        verbose: int = 1,
        random_batch: bool = True
    ) -> "WassersteinRegressor":
        """Train the Wasserstein regressor on given data.

        Args:
            X: Input features (numpy array or torch tensor).
            y: Target values (numpy array or torch tensor).
            epochs: Number of training epochs.
            lr: Learning rate for Adam optimizer.
            weight_decay: Weight decay coefficient.
            const_variance: Whether to use given constant variance (None = variance will be estimated for every point)
            batch_size: Batch size for training.
            verbose: Logging frequency (0=none, 1=some, >1=every N epochs).
            random_batch: Whether to use random batch selection.

        Returns:
            self: Trained model instance.
        """
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        X_norm = self._normalize(X)

        if const_variance is None:
            # Estimate local variance (subclass-specific)
            sigma_y = self.estimate_local_variance(X_norm, y)
        else:
            sigma_y = torch.full_like(y, const_variance)

        # Initialize control points (subclass-specific)
        self._initialize_control_points(X_norm, y, sigma_y)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5, weight_decay=weight_decay)

        for epoch in range(epochs):
            if random_batch:
                loss = self._fit_epoch_random_batch(X_norm, y, sigma_y, optimizer, batch_size=batch_size)
            else:
                loss = self._fit_epoch_full(X_norm, y, sigma_y, optimizer, batch_size=batch_size)

            if verbose > 0:
                log_message = epoch == epochs - 1
                log_message |= verbose == 1 and (epoch % max(1, epochs // 10) == 0)
                log_message |= verbose > 1 and epoch % verbose == 0
                if log_message:
                    print(f"Epoch {epoch:4d}, loss: {loss:.6f}")

        return self

    @torch.no_grad()
    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        with_std: bool = True,
        return_numpy: Optional[bool] = None
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor], np.ndarray, torch.Tensor]:
        """Predict mean and standard deviation for new inputs.

        Args:
            X: Input features (numpy array or torch tensor).
            with_std: Whether to return standard deviation alongside mean.
            return_numpy: Whether to return numpy arrays (None=auto-detect from X).

        Returns:
            Mean predictions, or (mean, std) tuple depending on with_std and return_numpy.
        """
        if self.mu_P is None or self.logvar_P is None:
            raise RuntimeError("Model not trained yet")

        # Convert to torch.Tensor, if return_numpy not provided set it to given X type
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            if return_numpy is None:
                return_numpy = True
        elif return_numpy is None:
            return_numpy = False

        X_norm = self._normalize(X)
        mu, sigma = self._evaluate(X_norm)

        if with_std and return_numpy:
            return mu.cpu().numpy(), torch.sqrt(sigma).cpu().numpy()
        elif with_std:
            return mu, torch.sqrt(sigma)
        elif return_numpy:
            return mu.cpu().numpy()
        else:
            return mu

    def __str__(self) -> str:
        """Return string representation of model state."""
        trained = self.mu_P is not None and self.logvar_P is not None
        desc = f"{self.__class__.__name__}[trained={trained}; device={self.device}"
        if trained:
            desc += f"; mu={tuple(self.mu_P.shape)}; logvar={tuple(self.logvar_P.shape)}"
        desc += "]"
        return desc


class WassersteinCurveRegressor(WassersteinRegressor):
    """Base class for 1-D curve regressors (Bezier, B-spline)."""

    def __init__(
        self,
        n_control_points: int,
        degree: int = 3,
        closed: bool = False,
        k_neighbors: int = 10,
        device: Optional[str] = None,
    ) -> None:
        """Initialize 1D curve regressor base.

        Args:
            n_control_points: Number of control points.
            degree: Spline degree.
            closed: Whether curve is closed/periodic.
            k_neighbors: Window size for local variance.
            device: Optional device string.
        """
        super().__init__(k_neighbors=k_neighbors, device=device)
        if n_control_points <= degree:
            raise ValueError("n_control_points must be greater than degree")
        self.n_control_points = int(n_control_points)
        self.degree = int(degree)
        self.closed = bool(closed)

    def estimate_local_variance(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimate per-sample variance using a sliding window of size ``k``."""
        sorted_idx = torch.argsort(X.squeeze())
        y_sorted = y[sorted_idx]
        pad = self.k // 2
        y_padded = torch.nn.functional.pad(
            y_sorted.unsqueeze(0).transpose(1, 2),
            (pad, pad),
            mode="replicate",
        ).transpose(1, 2).squeeze(0)
        windows = y_padded.unfold(0, self.k, 1).transpose(1, 2)
        var = windows.var(dim=1, unbiased=False) + 1e-6
        inv_idx = torch.argsort(sorted_idx)
        return var[inv_idx]

    def _initialize_control_points(
        self, X_norm: torch.Tensor, y: torch.Tensor, sigma_y: torch.Tensor
    ) -> None:
        """Initialize control points at evenly spaced indices."""
        n = y.shape[0]
        positions = np.linspace(0, n - 1, self.n_control_points, endpoint=not self.closed)
        init_idx = np.round(positions).astype(int) % n
        self.mu_P = nn.Parameter(y[init_idx].clone())
        self.logvar_P = nn.Parameter(torch.log(sigma_y[init_idx]).clone())

    @abstractmethod
    def _basis_matrix(self, x_unit: torch.Tensor) -> torch.Tensor:
        """Return spline basis matrix for curve evaluation."""

    def _evaluate(self, x_unit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate curve at normalized parameters."""
        basis = self._basis_matrix(x_unit)
        mu = basis @ self.mu_P
        logvar = basis @ self.logvar_P
        sigma = torch.exp(logvar)
        return mu, sigma


class WassersteinSurfaceRegressor(WassersteinRegressor):
    """Base class for 2-D surface regressors (Bezier, B-spline)."""

    def __init__(
        self,
        n_control_u: int,
        n_control_v: int,
        degree_u: int = 3,
        degree_v: int = 3,
        closed_u: bool = False,
        closed_v: bool = False,
        grid_size: int = 50,
        device: Optional[str] = None,
    ) -> None:
        """Initialize 2D surface regressor base.

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
        super().__init__(grid_size=grid_size, device=device)
        if n_control_u <= degree_u or n_control_v <= degree_v:
            raise ValueError("Control points must be greater than degree for each axis")
        self.Nu = int(n_control_u - 1)
        self.Nv = int(n_control_v - 1)
        self.degree_u = int(degree_u)
        self.degree_v = int(degree_v)
        self.closed_u = bool(closed_u)
        self.closed_v = bool(closed_v)

    def estimate_local_variance(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimate variance per sample using a 2-D histogram binning approach."""
        u = X[:, 0]
        v = X[:, 1]
        bins_u = torch.clamp((u * self.grid_size).long(), 0, self.grid_size - 1)
        bins_v = torch.clamp((v * self.grid_size).long(), 0, self.grid_size - 1)
        var = torch.zeros_like(y)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                mask = (bins_u == i) & (bins_v == j)
                if mask.sum() < 2:
                    continue
                local_y = y[mask]
                var[mask] = local_y.var(dim=0, unbiased=False)
        return var + 1e-6

    def _initialize_control_points(
        self, X_norm: torch.Tensor, y: torch.Tensor, sigma_y: torch.Tensor
    ) -> None:
        """Initialize control grid using nearest-point strategy."""
        grid_u = torch.linspace(0.0, 1.0, self.Nu + 1, device=self.device)
        grid_v = torch.linspace(0.0, 1.0, self.Nv + 1, device=self.device)
        mu_init = torch.zeros(self.Nu + 1, self.Nv + 1, y.shape[1], device=self.device)
        logvar_init = torch.zeros_like(mu_init)

        for i in range(self.Nu + 1):
            for j in range(self.Nv + 1):
                uv = torch.stack([grid_u[i], grid_v[j]])
                dist = torch.sum((X_norm - uv) ** 2, dim=1)
                idx = torch.argmin(dist)
                mu_init[i, j] = y[idx]
                logvar_init[i, j] = torch.log(sigma_y[idx])

        self.mu_P = nn.Parameter(mu_init.clone())
        self.logvar_P = nn.Parameter(logvar_init.clone())

    @abstractmethod
    def _basis_u(self, u_unit: torch.Tensor) -> torch.Tensor:
        """Return spline basis matrix for u direction."""

    @abstractmethod
    def _basis_v(self, v_unit: torch.Tensor) -> torch.Tensor:
        """Return spline basis matrix for v direction."""

    def _evaluate(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate surface at normalized (u,v) coordinates."""
        Bu = self._basis_u(X[:, 0])
        Bv = self._basis_v(X[:, 1])
        mu = torch.einsum("ni,ijc,nj->nc", Bu, self.mu_P, Bv)
        logvar = torch.einsum("ni,ijc,nj->nc", Bu, self.logvar_P, Bv)
        sigma = torch.exp(logvar)
        return mu, sigma
