"""
Generates the annotated "how to read a PSD panel" reference figure
(Figure 17): a single, real PSD panel from one reconstructed frame, with
extra annotations explaining the log-log axis convention and the physical
meaning of low vs. high wavenumber, so the dense multi-frame PSD rows in
Figures 23-25 are easier to interpret at a glance.

Reads directly from a .npz saved by compare_models.py -- no checkpoints, no
model code, no torch required.

Usage:
    python make_psd_reading_guide.py ../data/derived_npz/compare_lat-31.46_lon-119.01_start150_chunk5.npz
    python make_psd_reading_guide.py ../data/derived_npz/compare_lat-31.46_lon-119.01_start150_chunk5.npz \
        --frame 2 --out ../figures/fig17_psd_reading_guide.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--frame", type=int, default=2,
                     help="Interior frame index to plot (default: 2, an informative "
                          "withheld frame where all curves diverge)")
    ap.add_argument("--out", default="psd_reading_guide.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)
    t = args.frame

    k = d["psd_k"]
    psd_gt = d["psd_gt"][t]
    psd_interp = d["psd_interp"][t]
    psd_unet = d["psd_unet"][t]
    psd_diff_mean = d["psd_diffusion_mean"][t]
    psd_diff_std = d["psd_diffusion_std"][t]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    ax.loglog(k, psd_gt,        color="black", lw=2.2, label="Ground truth (GT)")
    ax.loglog(k, psd_interp,    color="red",   lw=1.6, linestyle=":",  label="Linear interpolation")
    ax.loglog(k, psd_unet,      color="green", lw=1.6, linestyle="--", label="U-Net reconstruction")
    ax.loglog(k, psd_diff_mean, color="blue",  lw=1.8, label="Diffusion ensemble mean")
    ax.fill_between(k, psd_diff_mean - psd_diff_std, psd_diff_mean + psd_diff_std,
                     color="blue", alpha=0.2, label="Diffusion ensemble ±1 std. dev.")

    ax.grid(True, which="major", ls="--", lw=0.5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", labelsize=13)

    ax.set_xlabel("Wavenumber $k$  (cycles per km, cpkm)", fontsize=14)
    ax.set_ylabel("Power spectral density\n(normalized SSHA$^2$ / cpkm)", fontsize=14)
    ax.set_title("How to read a PSD panel", fontsize=15, pad=14)

    ax.legend(fontsize=11.5, loc="lower left", frameon=True, framealpha=0.95)

    # Low-k vs high-k annotations and the log-log explanation box all live in
    # the empty middle-right region of the axes so nothing overlaps the
    # curves, legend, or title.
    ax.annotate(
        "Low $k$ = large spatial scales\n(coarse, slowly evolving features)",
        xy=(k[2], psd_gt[2]), xytext=(0.10, 0.65),
        textcoords="axes fraction", fontsize=11, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.2),
    )
    ax.annotate(
        "High $k$ = small spatial scales\n(fine structure, eddies/fronts)",
        xy=(k[-4], psd_gt[-4]), xytext=(0.55, 0.62),
        textcoords="axes fraction", fontsize=11, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.2),
    )
    ax.text(
        0.985, 0.985,
        "Log-log axes: equal distances = equal\nmultiplicative factors, not equal differences.\n"
        "Curves closer to GT (black) indicate\nbetter-preserved spectral energy at that scale.",
        transform=ax.transAxes, fontsize=9.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
