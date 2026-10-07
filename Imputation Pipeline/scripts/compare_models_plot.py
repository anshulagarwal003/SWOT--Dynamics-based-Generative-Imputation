"""
Plotting utility for compare_models.py output.

Rebuilds the UNet-vs-diffusion comparison figure (and each row as a separate
PNG) purely from the .npz saved by compare_models.py -- no checkpoints, no
model code, no torch required, so figures can be restyled without rerunning
inference.

Every chunk size is subset to exactly 5 representative days (evenly spaced,
always including both endpoints), so every figure uses the same layout
regardless of the reconstruction gap length:
    chunk_size=5  -> days 1,2,3,4,5   (all days)
    chunk_size=10 -> days 1,3,6,8,10
    chunk_size=21 -> days 1,6,11,16,21

Rows: ground truth, linear interpolation, U-Net reconstruction, diffusion
ensemble samples, isotropic PSD (with ensemble mean +/- std envelope), and
per-pixel CRPS. GT/Interp/UNet/Diffusion rows share one SSH colorbar; CRPS
has its own colorbar.

Usage:
    python compare_models_plot.py path/to/compare_output.npz
    python compare_models_plot.py path/to/compare_output.npz --out restyled.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator

# Shared left/right margins (figure-fraction) for the *panel columns* across
# every row type (dates, GT/Interp/UNet/Diffusion/CRPS heatmaps, PSD curves),
# so that when the separate row PNGs are stacked externally (e.g. in a LaTeX
# figure), each day's column lines up vertically across all rows. The
# colorbar (for heatmap rows) or wavenumber/PSD axis labels (for the PSD row)
# live to the right of RIGHT_MARGIN / left of LEFT_MARGIN respectively, and do
# not shift the panel columns themselves.
LEFT_MARGIN = 0.10
RIGHT_MARGIN = 0.90


N_FRAMES_SHOWN = 5  # every chunk size is subset down to exactly this many days


def select_day_indices(chunk_size):
    """Return the 0-indexed frame indices to display for a given chunk size.
    Every chunk size is subset to exactly N_FRAMES_SHOWN representative days,
    evenly spaced and always including both endpoints (1-indexed days
    converted to 0-indexed array positions). chunk_size == N_FRAMES_SHOWN
    (e.g. 5) trivially shows every frame."""
    if chunk_size == 10:
        days_1indexed = [1, 3, 6, 8, 10]
    elif chunk_size == 21:
        days_1indexed = [1, 6, 11, 16, 21]
    elif chunk_size <= N_FRAMES_SHOWN:
        return list(range(chunk_size))
    else:
        # generic fallback: evenly spaced days including both endpoints
        days_1indexed = [int(round(1 + i * (chunk_size - 1) / (N_FRAMES_SHOWN - 1)))
                          for i in range(N_FRAMES_SHOWN)]
    return [d - 1 for d in days_1indexed]


def save_row_figure(frames, dates, out_path, cmap, vmin, vmax, titles=None, dpi=300,
                     cbar_label=None):
    """Save one row (GT / Interp / UNet / Diff S1 / CRPS) as its own standalone
    figure. No per-panel date titles are added -- use save_dates_row_figure
    for a matching row of date labels. If cbar_label is given, a shared
    colorbar for this row's color scale is added on the right."""
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]
    im = None
    for t, ax in enumerate(axes):
        im = ax.imshow(frames[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[t], fontsize=15)
    plt.tight_layout(rect=[LEFT_MARGIN, 0, RIGHT_MARGIN, 1])
    if cbar_label:
        pos = axes[-1].get_position()
        cbar_ax = fig.add_axes([RIGHT_MARGIN + 0.02, pos.y0, 0.015, pos.y1 - pos.y0])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=15)
        cbar.ax.tick_params(labelsize=14)
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_dates_row_figure(dates, out_path, dpi=300):
    """Save a standalone image containing just one row of date labels, one per
    column, using the same per-column width (2.4 in) as save_row_figure so it
    lines up when stacked with the image rows in a document layout."""
    n = len(dates)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 0.6))
    if n == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.5, f"{dates[t]}", ha="center", va="center", fontsize=20,
                 transform=ax.transAxes)
    plt.tight_layout(rect=[LEFT_MARGIN, 0, RIGHT_MARGIN, 1])
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_psd_row_figure(psd_k, psd_gt, psd_interp, psd_unet, psd_diffusion_mean, psd_diffusion_std,
                         out_path, dpi=300):
    """Save the PSD row as its own standalone figure, with a shared y-axis,
    a top-mounted shared legend, and unit-labeled axes."""
    n = len(psd_gt)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.2))
    if n == 1:
        axes = [axes]
    legend_handles = None
    for t, ax in enumerate(axes):
        lines = []
        lines += ax.loglog(psd_k, psd_gt[t],     color="black", lw=1.8, label="GT")
        lines += ax.loglog(psd_k, psd_interp[t], color="red",   lw=1.3, linestyle=":",  label="Interp")
        lines += ax.loglog(psd_k, psd_unet[t],   color="green", lw=1.3, linestyle="--", label="UNet")
        lines += ax.loglog(psd_k, psd_diffusion_mean[t], color="blue", lw=1.5, label="Diffusion mean")
        band = ax.fill_between(psd_k, psd_diffusion_mean[t] - psd_diffusion_std[t],
                                psd_diffusion_mean[t] + psd_diffusion_std[t],
                                color="blue", alpha=0.2, label="Diffusion ±std")
        if t == 0:
            legend_handles = lines + [band]
        ax.grid(True, which="major", ls="--", lw=0.4)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", labelsize=17)

    fig.legend(handles=legend_handles, fontsize=14, loc="upper center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.2, handlelength=1.8, handletextpad=0.5)

    ymins, ymaxs = zip(*(ax.get_ylim() for ax in axes))
    shared_ylim = (min(ymins), max(ymaxs))
    for t, ax in enumerate(axes):
        ax.set_ylim(shared_ylim)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        if t != 0:
            ax.set_yticklabels([])

    plt.tight_layout(rect=[LEFT_MARGIN, 0.16, RIGHT_MARGIN, 0.85])
    plt.subplots_adjust(wspace=0.12, left=LEFT_MARGIN, right=RIGHT_MARGIN, bottom=0.24)

    left  = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    bottom = axes[0].get_position().y0
    top = axes[0].get_position().y1
    fig.text((left + right) / 2, bottom - 0.09, "wavenumber $k$ (cpkm)",
              ha="center", va="top", fontsize=15)

    fig.text(0.02, (bottom + top) / 2,
              "PSD (normalized SSHA$^2$ cpkm$^{-1}$)",
              ha="center", va="center", rotation=90, fontsize=12)

    plt.savefig(out_path, dpi=dpi)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <npz_path>_replot.png)")
    ap.add_argument("--rows-dpi", type=int, default=300, help="DPI for the separate per-row figures")
    args = ap.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)

    lat, lon           = float(d["lat"]), float(d["lon"])
    start               = int(d["start"])
    chunk_size          = int(d["chunk_size"])
    cold_start_t        = int(d["cold_start_t"])
    n_ensemble          = int(d["n_ensemble"])
    dates_full          = d["dates"]

    gt_frames_full     = d["gt_frames"]
    interp_frames_full = d["interp_frames"]
    unet_frames_full   = d["unet_frames"]
    ensemble_full      = d["diffusion_ensemble"]        # (N, T, H, W)

    crps_maps_full    = d["crps_maps"]                   # (T, H, W)
    crps_scalars_full = d["crps_scalars"]                # (T,)

    vmin, vmax, crps_max = float(d["vmin"]), float(d["vmax"]), float(d["crps_max"])

    psd_k              = d["psd_k"]
    psd_gt_full             = d["psd_gt"]
    psd_interp_full         = d["psd_interp"]
    psd_unet_full           = d["psd_unet"]
    psd_diffusion_mean_full = d["psd_diffusion_mean"]
    psd_diffusion_std_full  = d["psd_diffusion_std"]

    has_psd = psd_k.size > 0

    # subset to the representative days for this chunk size (all days for chunk_size=5)
    idx = select_day_indices(chunk_size)
    n_shown = len(idx)

    dates         = dates_full[idx]
    gt_frames     = gt_frames_full[idx]
    interp_frames = interp_frames_full[idx]
    unet_frames   = unet_frames_full[idx]
    ensemble      = ensemble_full[:, idx]

    crps_maps    = crps_maps_full[idx]
    crps_scalars = crps_scalars_full[idx]

    if has_psd:
        psd_gt             = psd_gt_full[idx]
        psd_interp         = psd_interp_full[idx]
        psd_unet           = psd_unet_full[idx]
        psd_diffusion_mean = psd_diffusion_mean_full[idx]
        psd_diffusion_std  = psd_diffusion_std_full[idx]

    n_ensemble_shown = 1  # only plot the first diffusion ensemble member (S1)

    gt_row, interp_row, unet_row = 0, 1, 2
    diff_rows = list(range(3, 3 + n_ensemble_shown))
    psd_row, crps_row = 3 + n_ensemble_shown, 4 + n_ensemble_shown
    n_rows = 5 + n_ensemble_shown

    fig, axes = plt.subplots(n_rows, n_shown, figsize=(2.2 * n_shown, 2 * n_rows))
    if n_shown == 1:
        axes = axes.reshape(-1, 1)

    ssh_im = None
    crps_im = None
    for t in range(n_shown):
        ssh_im = axes[gt_row,     t].imshow(gt_frames[t],     cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[interp_row, t].imshow(interp_frames[t], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[unet_row,   t].imshow(unet_frames[t],   cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for s, row in enumerate(diff_rows):
            axes[row, t].imshow(ensemble[s, t], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for row in (gt_row, interp_row, unet_row, *diff_rows):
            axes[row, t].set_xticks([]); axes[row, t].set_yticks([])

        ax_psd = axes[psd_row, t]
        if has_psd:
            ax_psd.loglog(psd_k, psd_gt[t],     color="black", lw=1.5, label="GT")
            ax_psd.loglog(psd_k, psd_interp[t], color="red",   lw=1.0, linestyle=":",  label="Interp")
            ax_psd.loglog(psd_k, psd_unet[t],   color="green", lw=1.0, linestyle="--", label="UNet")
            ax_psd.loglog(psd_k, psd_diffusion_mean[t], color="blue", lw=1.2, label="Diffusion mean")
            ax_psd.fill_between(psd_k, psd_diffusion_mean[t] - psd_diffusion_std[t],
                                 psd_diffusion_mean[t] + psd_diffusion_std[t],
                                 color="blue", alpha=0.2, label="Diffusion ±std")
        ax_psd.grid(True, which="both", ls="--", lw=0.4)
        ax_psd.tick_params(axis="both", labelsize=9)
        if t == 0:
            ax_psd.legend(fontsize=8, loc="lower left")

        crps_im = axes[crps_row, t].imshow(crps_maps[t], cmap="YlOrRd", vmin=0, vmax=crps_max)
        axes[crps_row, t].set_title(f"CRPS={crps_scalars[t]:.3f}", fontsize=9)
        axes[crps_row, t].set_xticks([]); axes[crps_row, t].set_yticks([])

    if has_psd:
        psd_axes = [axes[psd_row, t] for t in range(n_shown)]
        ymins, ymaxs = zip(*(ax.get_ylim() for ax in psd_axes))
        shared_ylim = (min(ymins), max(ymaxs))
        for t, ax in enumerate(psd_axes):
            ax.set_ylim(shared_ylim)
            if t != 0:
                ax.set_yticklabels([])

    axes[gt_row,     0].set_ylabel("GT",           fontsize=14, rotation=0, labelpad=35)
    axes[interp_row, 0].set_ylabel("Interp",       fontsize=14, rotation=0, labelpad=35)
    axes[unet_row,   0].set_ylabel("UNet",         fontsize=14, rotation=0, labelpad=35)
    for s, row in enumerate(diff_rows):
        axes[row, 0].set_ylabel(f"Diff S{s+1}",    fontsize=14, rotation=0, labelpad=35)
    axes[psd_row,    0].set_ylabel("PSD",          fontsize=14, rotation=0, labelpad=35)
    axes[crps_row,   0].set_ylabel("CRPS (diff.)", fontsize=14, rotation=0, labelpad=35)

    plt.suptitle(
        f"lat={lat:.3f}, lon={lon:.3f}, date=[{dates[0]}:{dates[-1]}]\n"
        f"UNet vs Diffusion (cold-start t={cold_start_t})",
        fontsize=15, y=0.998,
    )
    plt.tight_layout(rect=[0.07, 0, 0.83, 0.94])

    # shared SSH colorbar spanning the GT/Interp/UNet/Diffusion rows
    top_pos = axes[gt_row, -1].get_position()
    bottom_pos = axes[diff_rows[-1], -1].get_position()
    ssh_cbar_ax = fig.add_axes([0.85, bottom_pos.y0, 0.015, top_pos.y1 - bottom_pos.y0])
    ssh_cbar = fig.colorbar(ssh_im, cax=ssh_cbar_ax)
    ssh_cbar.set_label("SSHA (normalized)", fontsize=12)
    ssh_cbar.ax.tick_params(labelsize=11)

    # separate CRPS colorbar next to the CRPS row
    crps_pos = axes[crps_row, -1].get_position()
    crps_cbar_ax = fig.add_axes([0.85, crps_pos.y0, 0.015, crps_pos.y1 - crps_pos.y0])
    crps_cbar = fig.colorbar(crps_im, cax=crps_cbar_ax)
    crps_cbar.set_label("CRPS", fontsize=12)
    crps_cbar.ax.tick_params(labelsize=11)

    if has_psd:
        psd_axes = [axes[psd_row, t] for t in range(n_shown)]
        left  = psd_axes[0].get_position().x0
        right = psd_axes[-1].get_position().x1
        bottom = psd_axes[0].get_position().y0
        fig.text((left + right) / 2, bottom - 0.015, "wavenumber $k$",
                  ha="center", va="top", fontsize=12)

    out_path = args.out or args.npz_path.replace(".npz", "_replot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    base = os.path.splitext(out_path)[0]
    rows_dir = f"{base}_rows"
    os.makedirs(rows_dir, exist_ok=True)
    crps_titles = [f"CRPS={c:.3f}" for c in crps_scalars]
    save_dates_row_figure(dates, os.path.join(rows_dir, "row_dates.png"), dpi=args.rows_dpi)
    save_row_figure(gt_frames,     dates, os.path.join(rows_dir, "row_gt.png"),     "RdBu_r", vmin, vmax,
                     dpi=args.rows_dpi, cbar_label="SSHA (normalized)")
    save_row_figure(interp_frames, dates, os.path.join(rows_dir, "row_interp.png"), "RdBu_r", vmin, vmax,
                     dpi=args.rows_dpi, cbar_label="SSHA (normalized)")
    save_row_figure(unet_frames,   dates, os.path.join(rows_dir, "row_unet.png"),   "RdBu_r", vmin, vmax,
                     dpi=args.rows_dpi, cbar_label="SSHA (normalized)")
    for s in range(n_ensemble):
        save_row_figure(ensemble[s], dates, os.path.join(rows_dir, f"row_diff_s{s+1}.png"),
                         "RdBu_r", vmin, vmax, dpi=args.rows_dpi, cbar_label="SSHA (normalized)")
    save_row_figure(crps_maps,     dates, os.path.join(rows_dir, "row_crps.png"),   "YlOrRd", 0, crps_max,
                     titles=crps_titles, dpi=args.rows_dpi, cbar_label="CRPS")
    if has_psd:
        save_psd_row_figure(psd_k, psd_gt, psd_interp, psd_unet, psd_diffusion_mean, psd_diffusion_std,
                             os.path.join(rows_dir, "row_psd.png"), dpi=args.rows_dpi)


if __name__ == "__main__":
    main()
