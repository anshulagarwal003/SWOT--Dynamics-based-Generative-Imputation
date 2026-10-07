"""
Compare GT, linear interpolation, UNet prediction, and diffusion ensemble
(mean + PSD uncertainty band) for a specific lat/lon and time window.

Both checkpoints must share the same chunk_size (enforced) so GT/interp
windows line up frame-for-frame. Model hyperparameters (base, kT, and for
UNet also batch_size/lr) are parsed from each checkpoint filename.

Usage:
    python compare_models.py \
        --unet-ckpt local_checkpoints/chunk10_base16_kT3_bs128_lr1.00e-04_best_model.pt \
        --diffusion-ckpt local_checkpoints/ddpm_chunk10_base16_kT3_lr1.00e-04_best.pt \
        --lat -0.009682 --lon -11.010422 \
        --start 100 \
        --cold-start-t 700 --n-ensemble 10
"""

import argparse
import csv
import os
import re
from datetime import date, timedelta
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import UNET
from diffusion_code import DDPM, make_endpoints_condition, linear_interp, load_config, _crps_ensemble
from spectra import compute_isotropic_psd

UNET_RE = re.compile(r"chunk(?P<chunk_size>\d+)_base(?P<base>\d+)_kT(?P<kT>\d+)")

DATA_START_DATE = date(2011, 9, 13)


def index_to_date(idx):
    return (DATA_START_DATE + timedelta(days=int(idx))).isoformat()


DIFF_RE = re.compile(r"chunk(?P<chunk_size>\d+)_base(?P<base>\d+)_kT(?P<kT>\d+)")


def parse_unet_hparams(ckpt_path):
    m = UNET_RE.search(os.path.basename(ckpt_path))
    if not m:
        raise ValueError(f"Could not parse chunk_size/base/kT from UNet checkpoint filename: {ckpt_path}")
    return int(m["chunk_size"]), int(m["base"]), int(m["kT"])


def parse_diffusion_hparams(ckpt_path):
    m = DIFF_RE.search(os.path.basename(ckpt_path))
    if not m:
        raise ValueError(f"Could not parse chunk_size/base/kT from diffusion checkpoint filename: {ckpt_path}")
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
    ap.add_argument("--unet-ckpt", required=True)
    ap.add_argument("--diffusion-ckpt", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--t-dim", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--cold-start-t", type=int, default=700)
    ap.add_argument("--n-ensemble", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    unet_chunk, unet_base, unet_kT = parse_unet_hparams(args.unet_ckpt)
    diff_chunk, diff_base, diff_kT = parse_diffusion_hparams(args.diffusion_ckpt)

    if unet_chunk != diff_chunk:
        raise ValueError(
            f"chunk_size mismatch: UNet ckpt uses chunk_size={unet_chunk}, "
            f"diffusion ckpt uses chunk_size={diff_chunk}. Pick checkpoints with matching chunk_size."
        )
    chunk_size = unet_chunk
    print(f"UNet: base={unet_base}, kT={unet_kT} | Diffusion: base={diff_base}, kT={diff_kT} | chunk_size={chunk_size}")

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

    mask, x_cond = make_endpoints_condition(x0)
    interp = linear_interp(x0)

    # --- UNet prediction (single deterministic pass) ---
    unet_model = UNET(in_channels=1, out_channels=1, base=unet_base, kT=unet_kT).to(device)
    unet_model.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    unet_model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unet_pred = unet_model(x_cond)
        unet_pred = unet_pred * (1.0 - mask) + x0 * mask
    del unet_model

    # --- Diffusion ensemble ---
    diff_model = UNET(in_channels=3, out_channels=1, base=diff_base, t_dim=args.t_dim, kT=diff_kT).to(device)
    diff_model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device))
    diff_model.eval()
    ddpm = DDPM(timesteps=args.timesteps, device=str(device))

    ensemble = []
    with torch.no_grad():
        for s in range(args.n_ensemble):
            torch.manual_seed(s * 10)
            noise = torch.randn(x0.shape, device=device)
            t_tensor = torch.full((1,), args.cold_start_t, device=device, dtype=torch.long)
            x_init = ddpm.q_sample(interp, t_tensor, noise)
            pred = ddpm.sample(diff_model, x0.shape, x_cond, mask, x_init=x_init, start_t=args.cold_start_t)
            ensemble.append(pred[0, 0].cpu().numpy())
    del diff_model
    ensemble = np.stack(ensemble)  # (N, T, H, W)

    gt_frames     = x0[0, 0].cpu().numpy()
    interp_frames = interp[0, 0].cpu().numpy()
    unet_frames   = unet_pred[0, 0].cpu().numpy()

    vmin, vmax = gt_frames.min(), gt_frames.max()

    n_ensemble = args.n_ensemble
    gt_row, interp_row, unet_row = 0, 1, 2
    diff_rows = list(range(3, 3 + n_ensemble))
    psd_row, ratio_row, crps_row = 3 + n_ensemble, 4 + n_ensemble, 5 + n_ensemble
    n_rows = 6 + n_ensemble

    crps_maps, crps_scalars = [], []
    for t in range(chunk_size):
        cmap, cscalar = _crps_ensemble(gt_frames[t], ensemble[:, t])
        crps_maps.append(cmap)
        crps_scalars.append(cscalar)
    crps_max = max(m.max() for m in crps_maps) if crps_maps else 1.0

    fig, axes = plt.subplots(n_rows, chunk_size, figsize=(2.2 * chunk_size, 2 * n_rows))
    if chunk_size == 1:
        axes = axes.reshape(-1, 1)

    # per-frame PSD curves, stashed here so they can be saved to .npz below
    # (k axis assumed identical across frames/models; only kept once as k_ref)
    psd_gt_all, psd_interp_all, psd_unet_all = [], [], []
    psd_diff_mean_all, psd_diff_std_all = [], []
    ratio_interp_all, ratio_unet_all = [], []
    ratio_diff_mean_all, ratio_diff_std_all = [], []
    psd_k_ref = None

    for t in range(chunk_size):
        axes[gt_row, t].set_title(f"{index_to_date(args.start + t)}", fontsize=8)
        axes[gt_row,     t].imshow(gt_frames[t],     cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[interp_row, t].imshow(interp_frames[t], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[unet_row,   t].imshow(unet_frames[t],   cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for s, row in enumerate(diff_rows):
            axes[row, t].imshow(ensemble[s, t], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for row in (gt_row, interp_row, unet_row, *diff_rows):
            axes[row, t].set_xticks([]); axes[row, t].set_yticks([])

        k_gt, p_gt = compute_isotropic_psd(gt_frames[t], dx=2.0, dy=2.0)
        k_in, p_in = compute_isotropic_psd(interp_frames[t], dx=2.0, dy=2.0)
        k_un, p_un = compute_isotropic_psd(unet_frames[t], dx=2.0, dy=2.0)
        sample_psds = [compute_isotropic_psd(ensemble[s, t], dx=2.0, dy=2.0) for s in range(args.n_ensemble)]
        k_ref = next((k for k, p in sample_psds if k is not None), None)

        p_diff_stack = None
        if k_ref is not None:
            p_diff_stack = np.stack([p for k, p in sample_psds if p is not None])
            p_diff_mean = p_diff_stack.mean(axis=0)
            p_diff_std  = p_diff_stack.std(axis=0)
            if psd_k_ref is None:
                psd_k_ref = k_ref
            psd_diff_mean_all.append(p_diff_mean)
            psd_diff_std_all.append(p_diff_std)
        psd_gt_all.append(p_gt)
        psd_interp_all.append(p_in)
        psd_unet_all.append(p_un)

        ax_psd = axes[psd_row, t]
        if k_gt is not None:
            ax_psd.loglog(k_gt, p_gt, color="black", lw=1.5, label="GT")
            ax_psd.loglog(k_in, p_in, color="red", lw=1.0, linestyle=":", label="Interp")
            ax_psd.loglog(k_un, p_un, color="green", lw=1.0, linestyle="--", label="UNet")
            if k_ref is not None:
                ax_psd.loglog(k_ref, p_diff_mean, color="blue", lw=1.2, label="Diffusion mean")
                ax_psd.fill_between(k_ref, p_diff_mean - p_diff_std, p_diff_mean + p_diff_std,
                                     color="blue", alpha=0.2, label="Diffusion ±std")
        ax_psd.set_xticks([]); ax_psd.set_yticks([])
        ax_psd.grid(True, which="both", ls="--", lw=0.4)
        if t == 0:
            ax_psd.legend(fontsize=5, loc="lower left")

        ax_ratio = axes[ratio_row, t]
        if k_gt is not None:
            ratio_in = p_in / (p_gt + 1e-30)
            ratio_un = p_un / (p_gt + 1e-30)
            ratio_interp_all.append(ratio_in)
            ratio_unet_all.append(ratio_un)
            ax_ratio.semilogx(k_in, ratio_in, color="red", lw=1.0, linestyle=":", label="Interp/GT")
            ax_ratio.semilogx(k_un, ratio_un, color="green", lw=1.0, linestyle="--", label="UNet/GT")
            if k_ref is not None:
                ratio_mean = p_diff_mean / (p_gt + 1e-30)
                ratio_std  = p_diff_std / (p_gt + 1e-30)
                ratio_diff_mean_all.append(ratio_mean)
                ratio_diff_std_all.append(ratio_std)
                ax_ratio.semilogx(k_ref, ratio_mean, color="blue", lw=1.2, label="Diffusion/GT")
                ax_ratio.fill_between(k_ref, ratio_mean - ratio_std, ratio_mean + ratio_std,
                                       color="blue", alpha=0.2)
            ax_ratio.axhline(y=1.0, color="black", lw=1.0)
            ax_ratio.set_ylim(0, 3)
        ax_ratio.set_xticks([]); ax_ratio.set_yticks([])
        ax_ratio.grid(True, which="both", ls="--", lw=0.4)
        if t == 0:
            ax_ratio.legend(fontsize=5, loc="upper left")

        axes[crps_row, t].imshow(crps_maps[t], cmap="YlOrRd", vmin=0, vmax=crps_max)
        axes[crps_row, t].set_title(f"CRPS={crps_scalars[t]:.3f}", fontsize=6)
        axes[crps_row, t].set_xticks([]); axes[crps_row, t].set_yticks([])

    axes[gt_row,     0].set_ylabel("GT",           fontsize=9, rotation=0, labelpad=35)
    axes[interp_row, 0].set_ylabel("Interp",       fontsize=9, rotation=0, labelpad=35)
    axes[unet_row,   0].set_ylabel("UNet",         fontsize=9, rotation=0, labelpad=35)
    for s, row in enumerate(diff_rows):
        axes[row, 0].set_ylabel(f"Diff S{s+1}",    fontsize=9, rotation=0, labelpad=35)
    axes[psd_row,    0].set_ylabel("PSD",          fontsize=9, rotation=0, labelpad=35)
    axes[ratio_row,  0].set_ylabel("PSD ratio",    fontsize=9, rotation=0, labelpad=35)
    axes[crps_row,   0].set_ylabel("CRPS (diff.)", fontsize=9, rotation=0, labelpad=35)

    plt.suptitle(
        f"lat={args.lat:.3f}, lon={args.lon:.3f}, "
        f"date=[{index_to_date(args.start)}:{index_to_date(args.start + chunk_size - 1)}] "
        f"— UNet vs Diffusion (N={args.n_ensemble}, cold-start t={args.cold_start_t})",
        fontsize=10, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = args.out or f"compare_lat{args.lat:.2f}_lon{args.lon:.2f}_start{args.start}_chunk{chunk_size}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # save every array behind this figure to a .npz with the same basename,
    # so the plot can be restyled later without rerunning the models
    npz_path = os.path.splitext(out_path)[0] + ".npz"
    np.savez(
        npz_path,
        lat=args.lat, lon=args.lon, start=args.start, chunk_size=chunk_size,
        cold_start_t=args.cold_start_t, n_ensemble=args.n_ensemble,
        dates=np.array([index_to_date(args.start + t) for t in range(chunk_size)]),
        gt_frames=gt_frames, interp_frames=interp_frames, unet_frames=unet_frames,
        diffusion_ensemble=ensemble,                       # (N, T, H, W)
        crps_maps=np.stack(crps_maps), crps_scalars=np.array(crps_scalars),
        vmin=vmin, vmax=vmax, crps_max=crps_max,
        psd_k=psd_k_ref if psd_k_ref is not None else np.array([]),
        psd_gt=np.stack(psd_gt_all), psd_interp=np.stack(psd_interp_all), psd_unet=np.stack(psd_unet_all),
        psd_diffusion_mean=np.stack(psd_diff_mean_all) if psd_diff_mean_all else np.array([]),
        psd_diffusion_std=np.stack(psd_diff_std_all) if psd_diff_std_all else np.array([]),
        ratio_interp=np.stack(ratio_interp_all) if ratio_interp_all else np.array([]),
        ratio_unet=np.stack(ratio_unet_all) if ratio_unet_all else np.array([]),
        ratio_diffusion_mean=np.stack(ratio_diff_mean_all) if ratio_diff_mean_all else np.array([]),
        ratio_diffusion_std=np.stack(ratio_diff_std_all) if ratio_diff_std_all else np.array([]),
    )
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    main()
