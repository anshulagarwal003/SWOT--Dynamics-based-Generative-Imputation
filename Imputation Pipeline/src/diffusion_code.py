import os
import copy
import threading
#import math
import pprint
import numpy as np
import matplotlib.pyplot as plt

import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml

from model import UNET

from data import (
    load_individual_time_series,
    compute_per_series_splits,
    build_datasets_for_chunk_size,
    worker_init_fn,
)
from spectra import compute_isotropic_psd
from spectral_loss import spectral_loss
from skimage.metrics import structural_similarity as ssim


# ============================================================
# TIMESTEP EMBEDDING
# ============================================================

# def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
#     half = dim // 2
#     freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
#     args = t.float()[:, None] * freqs[None]
#     emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
#     if dim % 2 == 1:
#         emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
#     return emb


# ============================================================
# MODEL
# ============================================================

# def groupnorm(ch: int, max_groups: int = 8) -> nn.GroupNorm:
#     g = min(max_groups, ch)
#     while ch % g != 0:
#         g -= 1
#         if g == 1:
#             break
#     return nn.GroupNorm(g, ch)


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, t_dim, kT):
#         super().__init__()

#         padT = kT // 2
#         self.norm1 = groupnorm(in_channels)
#         self.conv1 = nn.Conv3d(
#             in_channels, out_channels,
#             kernel_size=(kT, 3, 3), padding=(padT, 1, 1),
#             padding_mode="replicate",
#         )
#         self.t_proj = nn.Linear(t_dim, out_channels)

#         self.norm2 = groupnorm(out_channels)
#         self.conv2 = nn.Conv3d(
#             out_channels, out_channels,
#             kernel_size=(kT, 3, 3), padding=(padT, 1, 1),
#             padding_mode="replicate",
#         )

#         self.skip = (
#             nn.Identity() if in_channels == out_channels
#             else nn.Conv3d(in_channels, out_channels, kernel_size=1)
#         )

#     def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
#         h = self.conv1(F.silu(self.norm1(x)))
#         h = h + self.t_proj(t_emb)[:, :, None, None, None]
#         h = self.conv2(F.silu(self.norm2(h)))
#         return h + self.skip(x)


# class DownsampleSpaceOnly(nn.Module):
#     def __init__(self, ch: int):
#         super().__init__()
#         self.conv = nn.Conv3d(
#             ch, ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
#             padding_mode="replicate",
#         )

#     def forward(self, x):
#         return self.conv(x)


# class UpsampleSpaceOnly(nn.Module):
#     def __init__(self, ch: int):
#         super().__init__()
#         self.conv = nn.Conv3d(ch, ch, kernel_size=3, padding=1, padding_mode="replicate")

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
#         return self.conv(x)


# class UNET(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, base=16, t_dim=128, kT=3):
#         super().__init__()
#         self.t_dim = t_dim
#         self.time_mlp = nn.Sequential(
#             nn.Linear(t_dim, t_dim * 4),
#             nn.SiLU(),
#             nn.Linear(t_dim * 4, t_dim),
#         )

#         self.in_conv = nn.Conv3d(in_channels, base, kernel_size=3, padding=1, padding_mode="replicate")

#         self.enc1  = DoubleConv(base,       base,     t_dim, kT=kT)
#         self.down1 = DownsampleSpaceOnly(base)

#         self.enc2  = DoubleConv(base,       base * 2, t_dim, kT=kT)
#         self.down2 = DownsampleSpaceOnly(base * 2)

#         self.enc3  = DoubleConv(base * 2,   base * 4, t_dim, kT=kT)
#         self.down3 = DownsampleSpaceOnly(base * 4)

#         self.mid1  = DoubleConv(base * 4,   base * 4, t_dim, kT=kT)
#         self.mid2  = DoubleConv(base * 4,   base * 4, t_dim, kT=kT)

#         self.up3   = UpsampleSpaceOnly(base * 4)
#         self.dec3  = DoubleConv(base * 8,   base * 4, t_dim, kT=kT)

#         self.up2   = UpsampleSpaceOnly(base * 4)
#         self.dec2  = DoubleConv(base * 6,   base * 2, t_dim, kT=kT)

#         self.up1   = UpsampleSpaceOnly(base * 2)
#         self.dec1  = DoubleConv(base * 3,   base,     t_dim, kT=kT)

#         self.out_norm = groupnorm(base)
#         self.out = nn.Conv3d(base, out_channels, kernel_size=1)

#     def forward(self, x_in, t):
#         t_emb = sinusoidal_timestep_embedding(t, self.t_dim)
#         t_emb = self.time_mlp(t_emb)

#         x = self.in_conv(x_in)

#         e1 = self.enc1(x, t_emb);  x = self.down1(e1)
#         e2 = self.enc2(x, t_emb);  x = self.down2(e2)
#         e3 = self.enc3(x, t_emb);  x = self.down3(e3)

#         x = self.mid1(x, t_emb)
#         x = self.mid2(x, t_emb)

#         x = self.up3(x);  x = self.dec3(torch.cat([x, e3], dim=1), t_emb)
#         x = self.up2(x);  x = self.dec2(torch.cat([x, e2], dim=1), t_emb)
#         x = self.up1(x);  x = self.dec1(torch.cat([x, e1], dim=1), t_emb)

#         return self.out(F.silu(self.out_norm(x)))


# ============================================================
# CONDITIONING
# ============================================================

def make_endpoints_condition(x0: torch.Tensor):
    B, C, T, H, W = x0.shape
    mask = torch.zeros((B, 1, T, H, W), device=x0.device, dtype=x0.dtype)
    mask[:, :, 0]  = 1.0
    mask[:, :, -1] = 1.0
    x_cond = x0 * mask
    return mask, x_cond


# ============================================================
# DDPM
# ============================================================

class DDPM:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.device = device
        self.T = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    def q_sample(self, x0, t, noise):
        s1 = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        s2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, x_cond, mask):
        inp = torch.cat([x_t, x_cond, mask], dim=1)
        eps = model(inp, t)

        beta_t      = self.betas[t][:, None, None, None, None]
        alpha_t     = self.alphas[t][:, None, None, None, None]
        alpha_bar_t = self.alphas_cumprod[t][:, None, None, None, None]

        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        x0_hat = torch.clamp(x0_hat, -6.0, 6.0)

        coef1 = torch.sqrt(self.alphas_cumprod_prev[t][:, None, None, None, None]) * beta_t / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - self.alphas_cumprod_prev[t][:, None, None, None, None]) / (1 - alpha_bar_t)
        mean  = coef1 * x0_hat + coef2 * x_t

        var   = self.posterior_variance[t][:, None, None, None, None]
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float()[:, None, None, None, None]
        x_prev = mean + nonzero * torch.sqrt(var) * noise

        x_prev = x_prev * (1 - mask) + x_cond
        return x_prev

    @torch.no_grad()
    def sample(self, model, shape, x_cond, mask, x_init=None, start_t=None):
        x = x_init if x_init is not None else torch.randn(shape, device=self.device)
        x = x * (1 - mask) + x_cond
        t_max = start_t if start_t is not None else self.T - 1
        for ti in reversed(range(t_max + 1)):
            t = torch.full((shape[0],), ti, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, x_cond, mask)
        return x


# ============================================================
# DDIM SAMPLER  (fast inference: ~50 steps instead of 1000)
# ============================================================

class DDIMSampler:
    def __init__(self, ddpm: DDPM, ddim_steps: int = 50, eta: float = 0.0):
        self.ddpm = ddpm
        self.eta  = eta

        # evenly spaced subset of timesteps
        step_ratio = ddpm.T // ddim_steps
        self.timesteps = list(reversed(range(0, ddpm.T, step_ratio)))[:ddim_steps]

    @torch.no_grad()
    def sample(self, model, shape, x_cond, mask, x_init=None):
        device = self.ddpm.device
        x = x_init if x_init is not None else torch.randn(shape, device=device)
        x = x * (1 - mask) + x_cond

        for i, ti in enumerate(self.timesteps):
            t      = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else 0

            inp = torch.cat([x, x_cond, mask], dim=1)
            eps = model(inp, t)

            alpha_bar      = self.ddpm.alphas_cumprod[ti]
            alpha_bar_prev = self.ddpm.alphas_cumprod[t_prev] if t_prev > 0 else torch.ones(1, device=device)

            x0_hat = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            x0_hat = torch.clamp(x0_hat, -6.0, 6.0)

            sigma = self.eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            noise  = torch.randn_like(x) if ti > 0 else torch.zeros_like(x)

            x = torch.sqrt(alpha_bar_prev) * x0_hat + dir_xt + sigma * noise
            x = x * (1 - mask) + x_cond

        return x


# ============================================================
# LOSS
# ============================================================

def linear_interp(x0):
    T = x0.shape[2]
    tau = torch.linspace(0, 1, T, device=x0.device, dtype=x0.dtype).view(1, 1, T, 1, 1)
    return (1.0 - tau) * x0[:, :, 0:1] + tau * x0[:, :, -1:]


def gradient_loss(pred, target, mask, loss_on_missing_only=True):
    dx_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    dx_tgt  = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    dy_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dy_tgt  = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    if loss_on_missing_only:
        wx = mask[:, :, :, :, 1:] + mask[:, :, :, :, :-1]
        wy = mask[:, :, :, 1:, :] + mask[:, :, :, :-1, :]
        wx = (wx == 0).float()
        wy = (wy == 0).float()
        lx = ((dx_pred - dx_tgt) ** 2 * wx).sum() / (wx.sum() + 1e-8)
        ly = ((dy_pred - dy_tgt) ** 2 * wy).sum() / (wy.sum() + 1e-8)
    else:
        lx = F.mse_loss(dx_pred, dx_tgt)
        ly = F.mse_loss(dy_pred, dy_tgt)
    return lx + ly


def ddpm_loss_eps(model, ddpm, x0, mask, x_cond, loss_on_missing_only=True, spectral_lambda=0.0):
    B  = x0.shape[0]
    t  = torch.randint(0, ddpm.T, (B,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)

    x_t = ddpm.q_sample(x0, t, noise)
    x_t = x_t * (1 - mask) + x_cond   # keep endpoints clean

    inp      = torch.cat([x_t, x_cond, mask], dim=1)
    eps_pred = model(inp, t)

    if loss_on_missing_only:
        w = 1 - mask
        eps_loss = ((eps_pred - noise) ** 2 * w).sum() / (w.sum() + 1e-8)
    else:
        eps_loss = F.mse_loss(eps_pred, noise)

    if spectral_lambda > 0:
        s1 = ddpm.sqrt_alphas_cumprod[t][:, None, None, None, None]
        s2 = ddpm.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        x0_hat = (x_t - s2 * eps_pred) / s1
        sp = spectral_loss(x0_hat, x0, mask=mask, sample_weight=ddpm.alphas_cumprod[t])
        return eps_loss + spectral_lambda * sp

    return eps_loss


# ============================================================
# TRAIN
# ============================================================

SAMPLER      = "ddim"  # overridden by run_sweep.py
COLD_START   = False   # if True, start inference from noisy interpolation
COLD_START_T = 200     # noise level for cold start
N_ENSEMBLE   = 3       # number of ensemble samples in visualization


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _crps_ensemble(gt, ensemble):
    """
    gt:       (H, W)
    ensemble: (N, H, W)
    returns:
        crps_map:    (H, W)  — per-pixel CRPS
        crps_scalar: float   — mean over all pixels
    """
    N = ensemble.shape[0]
    mae_term = np.mean(np.abs(ensemble - gt[None]), axis=0)        # (H, W)
    spread_term = np.zeros_like(mae_term)
    for i in range(N):
        for j in range(N):
            spread_term += np.abs(ensemble[i] - ensemble[j])
    spread_term /= (N * N)
    crps_map = mae_term - 0.5 * spread_term                        # (H, W)
    return crps_map, float(crps_map.mean())


def _viz_worker(vis_data, epoch, n_ensemble, cold_start_t, result_container):
    """CPU-only viz function — runs in background thread, appends wandb log dict to result_container."""
    vis_log = {}
    for vis_idx, d in enumerate(vis_data):
        gt_frames     = d["gt"]        # (T, H, W)
        interp_frames = d["interp"]    # (T, H, W)
        ensemble      = d["ensemble"]  # (N, T, H, W)
        meta          = d["meta"]      # {'file_name', 'start', 'end', 'lat', 'lon'}

        T_vis      = gt_frames.shape[0]
        vmin, vmax = gt_frames.min(), gt_frames.max()
        data_range = vmax - vmin

        mean_frames = ensemble.mean(axis=0)
        std_frames  = ensemble.std(axis=0)
        std_max     = std_frames.max() if std_frames.max() > 0 else 1.0

        ssim_interp_vals = [ssim(gt_frames[t], interp_frames[t], data_range=data_range) for t in range(T_vis)]
        ssim_mean_vals   = [ssim(gt_frames[t], mean_frames[t],   data_range=data_range) for t in range(T_vis)]

        # CRPS per frame
        crps_maps    = []
        crps_scalars = []
        for t in range(T_vis):
            cmap, cscalar = _crps_ensemble(gt_frames[t], ensemble[:, t])
            crps_maps.append(cmap)
            crps_scalars.append(cscalar)
            vis_log[f"crps/vis{vis_idx}_t{t}"] = cscalar

        vis_log[f"meta/vis{vis_idx}_lat"]   = meta["lat"]
        vis_log[f"meta/vis{vis_idx}_lon"]   = meta["lon"]
        vis_log[f"meta/vis{vis_idx}_start"] = meta["start"]
        vis_log[f"meta/vis{vis_idx}_end"]   = meta["end"]

        mean_row   = 1 + n_ensemble
        std_row    = 2 + n_ensemble
        interp_row = 3 + n_ensemble
        psd_row    = 4 + n_ensemble
        ratio_row  = 5 + n_ensemble
        ssim_row   = 6 + n_ensemble
        crps_row   = 7 + n_ensemble
        n_rows     = 8 + n_ensemble

        crps_max = max(m.max() for m in crps_maps) if crps_maps else 1.0

        fig, axes = plt.subplots(n_rows, T_vis, figsize=(2.2 * T_vis, 2 * n_rows))

        for t in range(T_vis):
            axes[0, t].set_title(f"t={t}", fontsize=8)
            axes[0, t].imshow(gt_frames[t], cmap="RdBu_r", vmin=vmin, vmax=vmax)

            for s in range(n_ensemble):
                axes[1 + s, t].imshow(ensemble[s, t], cmap="RdBu_r", vmin=vmin, vmax=vmax)

            axes[mean_row,   t].imshow(mean_frames[t],   cmap="RdBu_r", vmin=vmin,  vmax=vmax)
            axes[std_row,    t].imshow(std_frames[t],    cmap="hot",    vmin=0,      vmax=std_max)
            axes[interp_row, t].imshow(interp_frames[t], cmap="RdBu_r", vmin=vmin,  vmax=vmax)
            axes[crps_row,   t].imshow(crps_maps[t],     cmap="YlOrRd", vmin=0,     vmax=crps_max)
            axes[crps_row,   t].set_title(f"CRPS={crps_scalars[t]:.3f}", fontsize=6)

            for row in range(interp_row + 1):
                axes[row, t].set_xticks([]); axes[row, t].set_yticks([])
            axes[crps_row, t].set_xticks([]); axes[crps_row, t].set_yticks([])

            k_gt, p_gt = compute_isotropic_psd(gt_frames[t],     dx=2.0, dy=2.0)
            k_in, p_in = compute_isotropic_psd(interp_frames[t], dx=2.0, dy=2.0)
            # collect per-sample PSDs for both PSD and ratio plots
            sample_psds = []
            for s in range(n_ensemble):
                k_s, p_s = compute_isotropic_psd(ensemble[s, t], dx=2.0, dy=2.0)
                sample_psds.append((k_s, p_s))

            # use a colormap so each sample has a distinct visible color
            sample_colors = plt.cm.cool(np.linspace(0, 1, n_ensemble))

            ax_psd = axes[psd_row, t]
            if k_gt is not None:
                for s, (k_s, p_s) in enumerate(sample_psds):
                    if k_s is not None:
                        ax_psd.loglog(k_s, p_s, color=sample_colors[s], lw=0.7, alpha=0.7)
                ax_psd.loglog(k_gt, p_gt, color="black", lw=1.5, label="GT")
                ax_psd.loglog(k_in, p_in, color="red",   lw=1.0, linestyle=":", label="Interp")
                log_k = np.log10(k_gt)
                log_p = np.log10(p_gt + 1e-30)
                mid   = len(log_k) // 4
                slope, intercept = np.polyfit(log_k[mid:-mid], log_p[mid:-mid], 1)
                ref = 10 ** (intercept + slope * log_k)
                ax_psd.loglog(k_gt, ref, color="gray", lw=0.8, linestyle=":", label=f"slope={slope:.1f}")
            ax_psd.set_xticks([]); ax_psd.set_yticks([])
            ax_psd.grid(True, which="both", ls="--", lw=0.4)
            if t == 0:
                ax_psd.legend(fontsize=5, loc="lower left")

            ax_ratio = axes[ratio_row, t]
            if k_gt is not None and p_gt is not None:
                for s, (k_s, p_s) in enumerate(sample_psds):
                    if k_s is not None:
                        ax_ratio.semilogx(k_s, p_s / (p_gt + 1e-30),
                                          color=sample_colors[s], lw=0.7, alpha=0.7)
                ax_ratio.semilogx(k_gt, p_in / (p_gt + 1e-30), color="red", lw=1.0, linestyle=":", label="Interp/GT")
                ax_ratio.axhline(y=1.0, color="black", lw=1.0, linestyle="-")
                ax_ratio.set_ylim(0, 3)
            ax_ratio.set_xticks([]); ax_ratio.set_yticks([])
            ax_ratio.grid(True, which="both", ls="--", lw=0.4)
            if t == 0:
                ax_ratio.legend(fontsize=5, loc="upper left")

            ax_ssim = axes[ssim_row, t]
            ax_ssim.bar([0, 1], [ssim_mean_vals[t], ssim_interp_vals[t]], color=["blue", "red"], alpha=0.7)
            ax_ssim.axhline(y=1.0, color="black", lw=0.8, linestyle=":")
            ax_ssim.set_ylim(-0.1, 1.1)
            ax_ssim.set_xticks([0, 1])
            ax_ssim.set_xticklabels(["Mean", "Interp"], fontsize=6)
            ax_ssim.set_yticks([0, 0.5, 1.0])
            ax_ssim.tick_params(axis='y', labelsize=5)
            ax_ssim.grid(True, axis='y', ls="--", lw=0.4)

        axes[0, 0].set_ylabel("GT",        fontsize=9, rotation=0, labelpad=30)
        for s in range(n_ensemble):
            axes[1 + s, 0].set_ylabel(f"S{s+1}", fontsize=9, rotation=0, labelpad=30)
        axes[mean_row,   0].set_ylabel("Mean",      fontsize=9, rotation=0, labelpad=30)
        axes[std_row,    0].set_ylabel("Std",       fontsize=9, rotation=0, labelpad=30)
        axes[interp_row, 0].set_ylabel("Interp",    fontsize=9, rotation=0, labelpad=30)
        axes[psd_row,    0].set_ylabel("PSD",       fontsize=9, rotation=0, labelpad=30)
        axes[ratio_row,  0].set_ylabel("PSD ratio", fontsize=9, rotation=0, labelpad=30)
        axes[ssim_row,   0].set_ylabel("SSIM",      fontsize=9, rotation=0, labelpad=30)
        axes[crps_row,   0].set_ylabel("CRPS",      fontsize=9, rotation=0, labelpad=30)
        plt.suptitle(
            f"Epoch {epoch+1} — lat={meta['lat']:.2f}, lon={meta['lon']:.2f}, "
            f"t=[{meta['start']}:{meta['end']}] — cold-start ({n_ensemble} samples, t={cold_start_t})",
            fontsize=11
        )
        plt.tight_layout()
        vis_log[f"imputation_preview_idx{vis_idx}"] = wandb.Image(fig)
        plt.close(fig)

    result_container.append(vis_log)


def train(config=None):
    cfg = load_config()

    with wandb.init(config=config):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # Data
        data_folder  = cfg["data"]["folder"]
        global_mean  = np.float32(cfg["data"]["global_mean"])
        global_std   = np.float32(cfg["data"]["global_std"])

        time_series_list = load_individual_time_series(data_folder)

        train_dataset, val_dataset, test_dataset = build_datasets_for_chunk_size(
            time_series_list, config.chunk_size, global_mean, global_std,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )

        # large chunk_size needs a smaller batch_size to fit in GPU memory
        batch_size = 64 if config.chunk_size == 21 else config.batch_size
        print(f"chunk_size={config.chunk_size} -> using batch_size={batch_size}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn,
        )

        # Model + DDPM
        model = UNET(
            in_channels=3, out_channels=1,
            base=config.base, t_dim=config.t_dim, kT=config.kT,
        ).to(device)

        ddpm  = DDPM(timesteps=config.timesteps, device=str(device))
        if SAMPLER == "ddpm":
            ddim = ddpm  # use full DDPM reverse process (1000 steps)
            print("Using DDPM sampler (1000 steps)")
        else:
            ddim = DDIMSampler(ddpm, ddim_steps=50, eta=0.5)
            print("Using DDIM sampler (50 steps)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

        spectral_lambda = float(config.get("spectral_lambda", 0.0))
        print(f"Spectral loss lambda: {spectral_lambda}")

        checkpoint_dir = cfg["checkpoints"]["dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        run_name = f"ddpm_chunk{config.chunk_size}_base{config.base}_kT{config.kT}_lr{config.lr:.2e}"
        best_ckpt_path  = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
        final_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")

        best_val_loss   = float("inf")
        epochs_no_improve = 0
        early_stop_patience = 10
        viz_thread        = None
        vis_log_container = []  # shared between main and viz thread

        for epoch in range(config.epochs):
            # ── train ──────────────────────────────────────────
            model.train()
            train_loss_sum, train_seen = 0.0, 0

            for x0, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"):
                x0 = x0.to(device)
                B  = x0.shape[0]
                mask, x_cond = make_endpoints_condition(x0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ddpm_loss_eps(model, ddpm, x0, mask, x_cond,
                                         loss_on_missing_only=True,
                                         spectral_lambda=spectral_lambda)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                train_loss_sum += loss.item() * B
                train_seen     += B

            train_loss = train_loss_sum / max(train_seen, 1)

            # ── val (noise-prediction loss — fast, no sampling) ─
            model.eval()
            val_loss_sum, val_seen = 0.0, 0

            with torch.no_grad():
                for x0, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]", leave=False):
                    x0 = x0.to(device)
                    B  = x0.shape[0]
                    mask, x_cond = make_endpoints_condition(x0)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = ddpm_loss_eps(model, ddpm, x0, mask, x_cond,
                                             loss_on_missing_only=True,
                                             spectral_lambda=spectral_lambda)

                    val_loss_sum += loss.item() * B
                    val_seen     += B

            val_loss = val_loss_sum / max(val_seen, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(copy.deepcopy(model.state_dict()), best_ckpt_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

            print(
                f"Epoch {epoch+1:03d}/{config.epochs} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"best_val={best_val_loss:.5f} | no_improve={epochs_no_improve}/{early_stop_patience}"
            )

            # ── visualize every 5 epochs (GPU samples → background thread for plotting) ──
            if (epoch + 1) % 5 == 0:
                # wait for previous viz thread to finish before starting a new one
                if viz_thread is not None and viz_thread.is_alive():
                    print("Waiting for previous viz thread to finish...")
                    viz_thread.join()

                # check if previous thread produced wandb logs to upload
                if vis_log_container:
                    wandb.log(vis_log_container.pop())

                # collect GPU outputs (numpy arrays) on main thread
                snap_state = copy.deepcopy(model.state_dict())
                vis_data   = []  # list of (gt, ensemble, interp) numpy arrays per vis_idx

                snap_model = copy.deepcopy(model)
                snap_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
                snap_model.eval()

                with torch.no_grad():
                    for vis_idx in range(4):
                        x0_vis, meta_vis = test_dataset[vis_idx]
                        x0_vis = x0_vis.unsqueeze(0).to(device)
                        mask_vis, x_cond_vis = make_endpoints_condition(x0_vis)
                        interp_vis = linear_interp(x0_vis)

                        ensemble = []
                        for s in range(N_ENSEMBLE):
                            torch.manual_seed(s * 10 + vis_idx)
                            noise    = torch.randn(x0_vis.shape, device=device)
                            t_tensor = torch.full((x0_vis.shape[0],), COLD_START_T, device=device, dtype=torch.long)
                            x_init_s = ddpm.q_sample(interp_vis, t_tensor, noise)
                            pred_s   = ddpm.sample(snap_model, x0_vis.shape, x_cond_vis, mask_vis,
                                                   x_init=x_init_s, start_t=COLD_START_T)
                            ensemble.append(pred_s[0, 0].cpu().numpy())

                        vis_data.append({
                            "gt":     x0_vis[0, 0].cpu().numpy(),
                            "interp": interp_vis[0, 0].cpu().numpy(),
                            "ensemble": np.stack(ensemble),
                            "meta":   meta_vis,
                        })

                del snap_model  # free GPU memory

                # spawn background thread for plotting (CPU only)
                vis_log_container.clear()
                viz_thread = threading.Thread(
                    target=_viz_worker,
                    args=(vis_data, epoch, N_ENSEMBLE, COLD_START_T, vis_log_container),
                    daemon=True,
                )
                viz_thread.start()

            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        # flush final viz if thread is still running
        if viz_thread is not None and viz_thread.is_alive():
            print("Waiting for final viz thread...")
            viz_thread.join()
        if vis_log_container:
            wandb.log(vis_log_container.pop())

        torch.save(model.state_dict(), final_ckpt_path)
        print(f"Best checkpoint:  {best_ckpt_path}")
        print(f"Final checkpoint: {final_ckpt_path}")


# ============================================================
# SWEEP CONFIG + ENTRY POINT
# ============================================================

def main():
    wandb.login()

    sweep_config = {
        "method": "random",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "optimizer":   {"value": "adamW"},
            "base":        {"values": [8]},
            "epochs":      {"value": 100},
            "lr":          {"values": [1e-4, 3e-4]},
            "timesteps":   {"value": 1000},
            "batch_size":  {"values": [128, 64]},
            "kT":          {"values": [3, 5]},
            "t_dim":       {"value": 128},
            "chunk_size":  {"values": [5, 10]},
        },
    }
    pprint.pprint(sweep_config)

    cfg      = load_config()
    sweep_id = wandb.sweep(sweep_config, project=cfg["wandb"]["project"])
    print(f"Sweep ID: {sweep_id}")

    wandb.agent(sweep_id, function=train, count=1)


if __name__ == "__main__":
    main()
