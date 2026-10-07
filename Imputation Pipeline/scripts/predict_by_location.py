"""
Generate imputed frames for a specific lat/lon and time window, using a
trained checkpoint. Loads the real .npy file matching the requested lat/lon,
takes the requested [start:start+chunk_size] window as ground truth/anchors
(first and last frame are the known endpoints), runs an ensemble of cold-start
DDPM samples, and plots GT / ensemble members / mean / std / interp / PSD.

Model hyperparameters (base, kT, chunk_size) are parsed from the checkpoint
filename, e.g. ddpm_chunk21_base8_kT5_lr1.00e-04_best.pt

Usage:
    python predict_by_location.py \
        --ckpt local_checkpoints/ddpm_chunk21_base8_kT5_lr1.00e-04_best.pt \
        --lat -0.009682 --lon -11.010422 \
        --start 100 \
        --cold-start-t 700 --n-ensemble 10
"""

import argparse
import csv
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from model import UNET
from diffusion_code import DDPM, make_endpoints_condition, linear_interp, load_config, _crps_ensemble
from spectra import compute_isotropic_psd


CKPT_RE = re.compile(r"chunk(?P<chunk_size>\d+)_base(?P<base>\d+)_kT(?P<kT>\d+)")


def parse_ckpt_hparams(ckpt_path):
    m = CKPT_RE.search(os.path.basename(ckpt_path))
    if not m:
        raise ValueError(f"Could not parse chunk_size/base/kT from checkpoint filename: {ckpt_path}")
    return int(m["chunk_size"]), int(m["base"]), int(m["kT"])


def find_file_for_latlon(data_folder, lat, lon, csv_path="file_coordinates.csv", tol=1e-3):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_lat, f_lon = float(row["lat"]), float(row["lon"])
            if abs(f_lat - lat) < tol and abs(f_lon - lon) < tol:
                file_name = row["file_name"]
                return os.path.join(data_folder, file_name), file_name
    raise ValueError(f"No file found matching lat={lat}, lon={lon} (tol={tol}) in {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint path; chunk_size/base/kT parsed from filename")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", type=int, required=True, help="Start time index for the chunk window")
    ap.add_argument("--t-dim", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--cold-start-t", type=int, default=700)
    ap.add_argument("--n-ensemble", type=int, default=10)
    ap.add_argument("--out", default=None, help="Output image path (default: auto-named)")
    args = ap.parse_args()

    chunk_size, base, kT = parse_ckpt_hparams(args.ckpt)
    print(f"Parsed from checkpoint: chunk_size={chunk_size}, base={base}, kT={kT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    global_mean = np.float32(cfg["data"]["global_mean"])
    global_std  = np.float32(cfg["data"]["global_std"])

    file_path, file_name = find_file_for_latlon(cfg["data"]["folder"], args.lat, args.lon)
    print(f"Matched file: {file_name}")

    data = np.load(file_path, mmap_mode='r')
    T = data.shape[0]
    if args.start < 0 or args.start + chunk_size > T:
        raise ValueError(f"start={args.start} + chunk_size={chunk_size} exceeds series length T={T}")

    chunk = data[args.start:args.start + chunk_size].astype(np.float32)
    chunk = chunk - chunk.mean(axis=(1, 2), keepdims=True)
    chunk = (chunk - global_mean) / global_std
    x0 = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T, H, W)

    model = UNET(in_channels=3, out_channels=1, base=base, t_dim=args.t_dim, kT=kT).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ddpm = DDPM(timesteps=args.timesteps, device=str(device))

    mask, x_cond = make_endpoints_condition(x0)
    interp = linear_interp(x0)

    ensemble = []
    with torch.no_grad():
        for s in range(args.n_ensemble):
            torch.manual_seed(s * 10)
            noise = torch.randn(x0.shape, device=device)
            t_tensor = torch.full((1,), args.cold_start_t, device=device, dtype=torch.long)
            x_init = ddpm.q_sample(interp, t_tensor, noise)
            pred = ddpm.sample(model, x0.shape, x_cond, mask, x_init=x_init, start_t=args.cold_start_t)
            ensemble.append(pred[0, 0].cpu().numpy())
    ensemble = np.stack(ensemble)  # (N, T, H, W)

    gt_frames     = x0[0, 0].cpu().numpy()
    interp_frames = interp[0, 0].cpu().numpy()
    mean_frames   = ensemble.mean(axis=0)
    std_frames    = ensemble.std(axis=0)

    vmin, vmax = gt_frames.min(), gt_frames.max()
    std_max = std_frames.max() if std_frames.max() > 0 else 1.0
    n_ensemble = args.n_ensemble

    mean_row, std_row, interp_row, psd_row, ratio_row, crps_row = (
        1 + n_ensemble, 2 + n_ensemble, 3 + n_ensemble, 4 + n_ensemble, 5 + n_ensemble, 6 + n_ensemble,
    )
    n_rows = 7 + n_ensemble

    crps_maps    = []
    crps_scalars = []
    for t in range(chunk_size):
        cmap, cscalar = _crps_ensemble(gt_frames[t], ensemble[:, t])
        crps_maps.append(cmap)
        crps_scalars.append(cscalar)
    crps_max = max(m.max() for m in crps_maps) if crps_maps else 1.0

    fig, axes = plt.subplots(n_rows, chunk_size, figsize=(2.2 * chunk_size, 2 * n_rows))
    if chunk_size == 1:
        axes = axes.reshape(-1, 1)

    sample_colors = plt.cm.cool(np.linspace(0, 1, n_ensemble))

    for t in range(chunk_size):
        axes[0, t].set_title(f"idx={args.start + t}", fontsize=8)
        axes[0, t].imshow(gt_frames[t], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for s in range(n_ensemble):
            axes[1 + s, t].imshow(ensemble[s, t], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[mean_row,   t].imshow(mean_frames[t],   cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[std_row,    t].imshow(std_frames[t],    cmap="hot",    vmin=0,    vmax=std_max)
        axes[interp_row, t].imshow(interp_frames[t], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[crps_row,   t].imshow(crps_maps[t], cmap="YlOrRd", vmin=0, vmax=crps_max)
        axes[crps_row,   t].set_title(f"CRPS={crps_scalars[t]:.3f}", fontsize=6)
        for row in range(interp_row + 1):
            axes[row, t].set_xticks([]); axes[row, t].set_yticks([])
        axes[crps_row, t].set_xticks([]); axes[crps_row, t].set_yticks([])

        k_gt, p_gt = compute_isotropic_psd(gt_frames[t], dx=2.0, dy=2.0)
        k_in, p_in = compute_isotropic_psd(interp_frames[t], dx=2.0, dy=2.0)
        sample_psds = [compute_isotropic_psd(ensemble[s, t], dx=2.0, dy=2.0) for s in range(n_ensemble)]

        ax_psd = axes[psd_row, t]
        if k_gt is not None:
            for s, (k_s, p_s) in enumerate(sample_psds):
                if k_s is not None:
                    ax_psd.loglog(k_s, p_s, color=sample_colors[s], lw=0.7, alpha=0.7)
            ax_psd.loglog(k_gt, p_gt, color="black", lw=1.5, label="GT")
            ax_psd.loglog(k_in, p_in, color="red", lw=1.0, linestyle=":", label="Interp")
        ax_psd.set_xticks([]); ax_psd.set_yticks([])
        ax_psd.grid(True, which="both", ls="--", lw=0.4)
        if t == 0:
            ax_psd.legend(fontsize=5, loc="lower left")

        ax_ratio = axes[ratio_row, t]
        if k_gt is not None:
            for s, (k_s, p_s) in enumerate(sample_psds):
                if k_s is not None:
                    ax_ratio.semilogx(k_s, p_s / (p_gt + 1e-30), color=sample_colors[s], lw=0.7, alpha=0.7)
            ax_ratio.semilogx(k_gt, p_in / (p_gt + 1e-30), color="red", lw=1.0, linestyle=":", label="Interp/GT")
            ax_ratio.axhline(y=1.0, color="black", lw=1.0)
            ax_ratio.set_ylim(0, 3)
        ax_ratio.set_xticks([]); ax_ratio.set_yticks([])
        ax_ratio.grid(True, which="both", ls="--", lw=0.4)
        if t == 0:
            ax_ratio.legend(fontsize=5, loc="upper left")

    axes[0, 0].set_ylabel("GT", fontsize=9, rotation=0, labelpad=30)
    for s in range(n_ensemble):
        axes[1 + s, 0].set_ylabel(f"S{s+1}", fontsize=9, rotation=0, labelpad=30)
    axes[mean_row,   0].set_ylabel("Mean",      fontsize=9, rotation=0, labelpad=30)
    axes[std_row,    0].set_ylabel("Std",       fontsize=9, rotation=0, labelpad=30)
    axes[interp_row, 0].set_ylabel("Interp",    fontsize=9, rotation=0, labelpad=30)
    axes[psd_row,    0].set_ylabel("PSD",       fontsize=9, rotation=0, labelpad=30)
    axes[ratio_row,  0].set_ylabel("PSD ratio", fontsize=9, rotation=0, labelpad=30)
    axes[crps_row,   0].set_ylabel("CRPS",      fontsize=9, rotation=0, labelpad=30)

    plt.suptitle(
        f"lat={args.lat:.3f}, lon={args.lon:.3f}, idx=[{args.start}:{args.start + chunk_size}] "
        f"— {os.path.basename(args.ckpt)} — cold-start t={args.cold_start_t}, N={n_ensemble}",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = args.out or f"pred_lat{args.lat:.2f}_lon{args.lon:.2f}_start{args.start}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
