"""
Plug-and-play isotropic spectral loss for diffusion training.

Matches the radially averaged (isotropic) log-PSD of the model's implied
clean estimate x0_hat against ground truth, per frame — the same statistic
plotted in the PSD/PSD-ratio viz rows. It constrains energy per wavenumber
band, not individual Fourier modes, so the model stays free to generate a
different realization with the right spectrum.

Disabled entirely when lambda = 0 (no FFT is ever computed).
"""

import torch

# cache of (bin_index, bin_count) per (H, W, device)
_BIN_CACHE = {}


def _radial_bins(H, W, device):
    key = (H, W, str(device))
    if key in _BIN_CACHE:
        return _BIN_CACHE[key]

    Wf = W // 2 + 1
    ky = torch.fft.fftfreq(H, device=device)          # (H,)
    kx = torch.fft.rfftfreq(W, device=device)         # (Wf,)
    k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)  # (H, Wf)

    n_bins = min(H, W) // 2
    k_max = 0.5  # Nyquist for unit spacing
    idx = torch.clamp((k / k_max * n_bins).long(), max=n_bins - 1).reshape(-1)  # (H*Wf,)
    counts = torch.bincount(idx, minlength=n_bins).float().clamp(min=1.0)       # (n_bins,)

    _BIN_CACHE[key] = (idx, counts, n_bins)
    return _BIN_CACHE[key]


def _isotropic_log_psd(x):
    """x: (B, C, T, H, W) -> radially averaged log-PSD (B*C*T, n_bins)."""
    B, C, T, H, W = x.shape
    x = x.float()  # FFT under bfloat16 autocast is unreliable; force fp32
    F = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    P = (F.real ** 2 + F.imag ** 2).reshape(B * C * T, -1)  # (N, H*Wf)

    idx, counts, n_bins = _radial_bins(H, W, x.device)
    psd = torch.zeros(P.shape[0], n_bins, device=x.device)
    psd.scatter_add_(1, idx[None, :].expand(P.shape[0], -1), P)
    psd = psd / counts[None, :]
    return torch.log(psd + 1e-12)


def spectral_loss(x0_hat, x0, mask=None, sample_weight=None):
    """
    x0_hat:        (B, C, T, H, W) model's implied clean estimate
    x0:            (B, C, T, H, W) ground truth
    mask:          (B, 1, T, H, W) 1 on known (endpoint) frames — those are excluded
    sample_weight: (B,) per-sample weight, e.g. alpha_bar_t, to downweight
                   high-t steps where x0_hat is estimated from near-pure noise
    """
    B, C, T = x0.shape[:3]
    lp_pred = _isotropic_log_psd(x0_hat)   # (B*C*T, n_bins)
    lp_gt   = _isotropic_log_psd(x0)
    diff = ((lp_pred - lp_gt) ** 2).mean(dim=-1).reshape(B, C, T)  # (B, C, T)

    if mask is not None:
        frame_w = 1.0 - mask[:, :, :, 0, 0]            # (B, 1, T)
        per_sample = (diff * frame_w).sum(dim=(1, 2)) / (frame_w.sum(dim=(1, 2)) + 1e-8)
    else:
        per_sample = diff.mean(dim=(1, 2))

    if sample_weight is not None:
        return (per_sample * sample_weight).mean()
    return per_sample.mean()
