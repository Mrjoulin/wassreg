from math import comb
import torch


def bernstein_basis(t, N: int):
    """Compute Bernstein basis functions.

    Args:
        t: Tensor of shape (M,) with values in [0,1].
        N: Bernstein degree (N+1 basis functions).

    Returns:
        Tensor of shape (M, N+1) with basis values.
    """
    t = t.view(-1, 1)
    powers = torch.arange(N + 1, device=t.device)
    binoms = torch.tensor(
        [comb(N, i) for i in range(N + 1)],
        dtype=torch.float32,
        device=t.device
    )
    B = binoms * (t ** powers) * ((1 - t) ** (N - powers))
    return B


def bspline_basis(t, n_control_points: int, degree: int, closed: bool):
    """Dispatch to open or closed B-spline basis.

    Args:
        t: Tensor of shape (M,) with values in [0,1].
        n_control_points: Number of control points.
        degree: B-spline degree.
        closed: Whether to use closed (periodic) basis.

    Returns:
        Tensor of shape (M, n_control_points) with basis values.
    """
    if closed:
        return closed_bspline_basis(t, n_control_points, degree)
    else:
        return open_bspline_basis(t, n_control_points, degree)


def open_bspline_basis(t, n_control_points: int, degree: int):
    """
    Compute open uniform B-spline basis functions.

    Args:
        t: Tensor of shape (N,) values in [0, 1] (will be clamped).
        n_control_points: Number of control points (>= degree+1).
        degree: B-spline degree.

    Returns:
        Tensor of shape (N, n_control_points) where each row sums to 1.
    """
    if n_control_points <= degree:
        raise ValueError("n_control_points must be greater than degree")

    # Ensure t is a 1D float tensor
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1)
    t = t.clamp(0.0, 1.0)

    knots = _open_uniform_knots(n_control_points, degree, device=t.device, dtype=t.dtype)
    basis = _bspline_basis(t, knots, degree, n_control_points)
    return _normalize_rows(basis)


def closed_bspline_basis(t, n_control_points: int, degree: int):
    """
    Compute closed (periodic) uniform B-spline basis functions.

    Args:
        t: Tensor of shape (N,) values (will be taken modulo 1).
        n_control_points: Number of control points (>= degree+1).
        degree: B-spline degree.

    Returns:
        Tensor of shape (N, n_control_points) where each row sums to 1.
    """
    if n_control_points <= degree:
        raise ValueError("n_control_points must be greater than degree")

    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1)
    # Map to [0, n_control_points) with wrap-around
    t_scaled = torch.remainder(t, 1.0) * n_control_points
    # Avoid t exactly equal to n_control_points
    t_scaled = torch.where(
        t_scaled == n_control_points,
        torch.nextafter(
            torch.tensor(float(n_control_points), device=t.device),
            torch.tensor(0.0, device=t.device)
        ),
        t_scaled
    )

    coeff_count = n_control_points + degree
    knots = torch.arange(-degree, n_control_points + degree + 1,
                         dtype=torch.float32, device=t.device)

    # Compute extended basis of size coeff_count
    extended = _bspline_basis(t_scaled, knots, degree, coeff_count)

    # Wrap and sum (periodic summation)
    basis = torch.zeros(len(t), n_control_points, dtype=t.dtype, device=t.device)
    for i in range(coeff_count):
        basis[:, i % n_control_points] += extended[:, i]

    return _normalize_rows(basis)


def _open_uniform_knots(n_control_points, degree, device, dtype):
    """Build open uniform knot vector on [0,1]."""
    inner_count = n_control_points - degree - 1
    if inner_count > 0:
        inner = torch.linspace(0.0, 1.0, inner_count + 2, dtype=dtype, device=device)[1:-1]
    else:
        inner = torch.empty(0, dtype=dtype, device=device)
    knots = torch.cat([
        torch.zeros(degree + 1, dtype=dtype, device=device),
        inner,
        torch.ones(degree + 1, dtype=dtype, device=device)
    ])
    return knots


def _bspline_basis(t, knots, degree, n_basis):
    """
    Evaluate B-spline basis of given degree using the Cox-de Boor recursion with vectorization.

    Args:
        t: Tensor of shape (N,) evaluation points.
        knots: Tensor of shape (n_basis + degree + 1,) uniform or open knots.
        degree: Spline degree.
        n_basis: Number of basis functions.

    Returns:
        Tensor of shape (N, n_basis) with the basis values.
    """
    N = len(t)
    device = t.device
    dtype = t.dtype

    # 1. Find knot span indices
    # s such that knots[s] <= t_i < knots[s+1]
    # Using side='right' gives insertion point after last occurrence of t_i
    s_raw = torch.searchsorted(knots, t, side='right') - 1
    # Clamp to valid range: p <= s <= n_basis-1
    s = torch.clamp(s_raw, min=degree, max=n_basis - 1)

    # 2. Precompute left and right arrays for all j=1..p
    # left_arr[:, j] = t - knots[s + 1 - j]
    # right_arr[:, j] = knots[s + j] - t
    # We'll store shape (N, p+1) with index 0 unused (set to 0)
    left_arr = torch.zeros(N, degree + 1, dtype=dtype, device=device)
    right_arr = torch.zeros(N, degree + 1, dtype=dtype, device=device)
    for j in range(1, degree + 1):
        # Indices into knots: s + 1 - j and s + j
        idx_left = s + 1 - j
        idx_right = s + j
        left_arr[:, j] = t - knots[idx_left]
        right_arr[:, j] = knots[idx_right] - t

    # 3. Cox‑de Boor recursion (batched over N)
    # N_vals holds the local basis functions N_{r}(t) for r = 0..p
    N_vals = torch.zeros(N, degree + 1, dtype=dtype, device=device)
    N_vals[:, 0] = 1.0

    for j in range(1, degree + 1):
        saved = torch.zeros(N, dtype=dtype, device=device)
        for r in range(j):
            denom = right_arr[:, r + 1] + left_arr[:, j - r]
            # Avoid division by zero: set temp = 0 where denom <= 0
            # (denom > 0 strictly inside knot spans; at boundaries some are zero)
            safe_denom = denom.clone()
            safe_denom[safe_denom <= 0.0] = 1.0  # dummy value, will be masked
            temp = N_vals[:, r] / safe_denom
            temp = torch.where(denom > 0.0, temp, torch.zeros_like(temp))
            N_vals[:, r] = saved + right_arr[:, r + 1] * temp
            saved = left_arr[:, j - r] * temp
        N_vals[:, j] = saved

    # 4. Scatter local values to global basis indices
    # Global index = s - degree + r
    global_idx = s.unsqueeze(1) - degree + torch.arange(degree + 1, device=device).unsqueeze(0)
    # Ensure indices are within [0, n_basis-1] (should hold by construction)
    basis = torch.zeros(N, n_basis, dtype=dtype, device=device)
    basis.scatter_add_(1, global_idx, N_vals)
    return basis


def _normalize_rows(basis):
    """Normalize each row to sum to 1, zero rows become uniform."""
    basis = torch.clamp(basis, min=0.0)
    row_sums = basis.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums <= 0.0, torch.ones_like(row_sums), row_sums)
    return basis / row_sums


def tensor_product_basis(basis_u, basis_v):
    """Compute tensor product of two basis matrices for 2D surfaces.

    Args:
        basis_u: Tensor (M, Nu) or array-like, u-direction basis.
        basis_v: Tensor (M, Nv) or array-like, v-direction basis.

    Returns:
        Tensor of shape (M, Nu*Nv) with flattened tensor product basis.
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(basis_u, torch.Tensor):
        basis_u = torch.as_tensor(basis_u, dtype=torch.float32)
    if not isinstance(basis_v, torch.Tensor):
        basis_v = torch.as_tensor(basis_v, dtype=torch.float32)
    if basis_u.shape[0] != basis_v.shape[0]:
        raise ValueError("basis_u and basis_v must have the same number of rows")
    return torch.einsum("ni,nj->nij", basis_u, basis_v).reshape(basis_u.shape[0], -1)
