"""
Inference-only test: does raising COLD_START_T recover high-k PSD energy
on an already-trained checkpoint? No training involved.

Usage:
    python evaluate_cold_start.py \
        --ckpt local_checkpoints/ddpm_chunk5_base16_kT3_lr3.00e-04_best.pt \
        --base 16 --kT 3 --chunk-size 5 --t-dim 128 \
        --cold-start-ts 750 850 900 950 \
        --n-ensemble 5 --n-vis 4
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import UNET
from diffusion_code import DDPM, make_endpoints_condition, linear_interp, load_config
from data import load_individual_time_series, compute_per_series_splits, build_datasets_for_chunk_size, worker_init_fn
from spectra import compute_isotropic_psd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--kT", type=int, required=True)
    ap.add_argument("--t-dim", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--cold-start-ts", type=int, nargs="+", default=[750, 850, 900, 950])
    ap.add_argument("--n-ensemble", type=int, default=5)
    ap.add_argument("--n-vis", type=int, default=4)
    ap.add_argument("--out", default="cold_start_sweep.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()

    global_mean = np.float32(cfg["data"]["global_mean"])
    global_std  = np.float32(cfg["data"]["global_std"])

    time_series_list  = load_individual_time_series(cfg["data"]["folder"])
    per_series_splits = compute_per_series_splits(
        time_series_list,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )
    _, _, test_dataset = build_datasets_for_chunk_size(
        per_series_splits, args.chunk_size, global_mean, global_std
    )

    model = UNET(in_channels=3, out_channels=1, base=args.base, t_dim=args.t_dim, kT=args.kT).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ddpm = DDPM(timesteps=args.timesteps, device=str(device))

    # rows = cold_start_ts, cols = n_vis samples' PSD-ratio plots
    fig, axes = plt.subplots(len(args.cold_start_ts), args.n_vis,
                              figsize=(3.2 * args.n_vis, 2.6 * len(args.cold_start_ts)),
                              squeeze=False)

    with torch.no_grad():
        for vis_idx in range(args.n_vis):
            x0_vis, _ = test_dataset[vis_idx]
            x0_vis = x0_vis.unsqueeze(0).to(device)
            mask_vis, x_cond_vis = make_endpoints_condition(x0_vis)
            interp_vis = linear_interp(x0_vis)

            T_vis = x0_vis.shape[2]
            mid_t = T_vis // 2  # a middle (imputed) frame — most affected by cold start

            gt_frame = x0_vis[0, 0, mid_t].cpu().numpy()
            k_gt, p_gt = compute_isotropic_psd(gt_frame, dx=2.0, dy=2.0)

            for row, cst in enumerate(args.cold_start_ts):
                ensemble = []
                for s in range(args.n_ensemble):
                    torch.manual_seed(s * 10 + vis_idx)
                    noise = torch.randn(x0_vis.shape, device=device)
                    t_tensor = torch.full((x0_vis.shape[0],), cst, device=device, dtype=torch.long)
                    x_init = ddpm.q_sample(interp_vis, t_tensor, noise)
                    pred = ddpm.sample(model, x0_vis.shape, x_cond_vis, mask_vis, x_init=x_init, start_t=cst)
                    ensemble.append(pred[0, 0, mid_t].cpu().numpy())

                ax = axes[row, vis_idx]
                sample_colors = plt.cm.cool(np.linspace(0, 1, args.n_ensemble))
                for s, frame in enumerate(ensemble):
                    k_s, p_s = compute_isotropic_psd(frame, dx=2.0, dy=2.0)
                    if k_s is not None:
                        ax.semilogx(k_s, p_s / (p_gt + 1e-30), color=sample_colors[s], lw=0.8, alpha=0.8)
                ax.axhline(y=1.0, color="black", lw=1.0)
                ax.set_ylim(0, 3)
                ax.grid(True, which="both", ls="--", lw=0.4)
                if vis_idx == 0:
                    ax.set_ylabel(f"t={cst}", fontsize=10)
                if row == 0:
                    ax.set_title(f"vis {vis_idx} (frame {mid_t})", fontsize=9)

    plt.suptitle(f"PSD ratio (sample/GT) vs cold-start t — ckpt: {args.ckpt}", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
