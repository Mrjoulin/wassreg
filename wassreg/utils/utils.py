import torch


def w2_diag_loss(mu_t, sigma_t, y, sigma_y):
    """Compute diagonal Wasserstein-2 loss between two Gaussian diagonals.

    Args:
        mu_t: Predicted mean tensor (batch, ...).
        sigma_t: Predicted variance tensor (batch, ...).
        y: Target mean tensor (batch, ...).
        sigma_y: Target variance tensor (batch, ...).

    Returns:
        Scalar mean loss.
    """
    loss = torch.sum((mu_t - y) ** 2, dim=1)
    loss += torch.sum((torch.sqrt(sigma_t) - torch.sqrt(sigma_y)) ** 2, dim=1)
    return loss.mean()


def get_device(device: str = None):
    """Select the best available torch device.

    Args:
        device: Optional device string (e.g., "cuda", "cpu"). Auto-detects if None.

    Returns:
        torch.device object.
    """
    if device is not None:
        return torch.device(device)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
