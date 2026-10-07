"""
Per-frame CRPS as a line plot over the chunk's time axis.

CRPS(t) is read directly from crps_scalars (already computed per frame; zero
at the observed endpoint frames).

Reads directly from the .npz saved by compare_models.py -- no checkpoints,
no model code, no torch required.

Usage:
    python plot_crps_timeseries.py compare_lat-31.46_lon-119.01_start150_chunk5.npz
    python plot_crps_timeseries.py compare_lat-31.46_lon-119.01_start150_chunk5.npz --out custom.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <npz_path>_crps_timeseries.png)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)

    lat, lon = float(d["lat"]), float(d["lon"])
    chunk_size = int(d["chunk_size"])
    dates = d["dates"]

    crps_scalars = d["crps_scalars"]           # (T,)
    T = crps_scalars.shape[0]

    t_axis = np.arange(T)

    # fixed figure size regardless of chunk length, so chunk5/10/21 plots are
    # directly comparable in a document layout; longer chunks instead thin out
    # which x-tick labels are shown (every 2nd or 3rd day) to avoid crowding
    fig, ax_crps = plt.subplots(figsize=(9, 5.5))

    ax_crps.plot(t_axis, crps_scalars, marker="o", color="firebrick", lw=1.8, markersize=8)
    ax_crps.set_ylabel("CRPS", fontsize=15)
    ax_crps.set_xlabel("Day", fontsize=15, labelpad=15)
    ax_crps.grid(True, ls="--", lw=0.4)
    ax_crps.tick_params(axis="y", labelsize=13)
    fig.suptitle(
        f"lat={lat:.3f}, lon={lon:.3f}, chunk_size={chunk_size}\n"
        f"Per-frame CRPS",
        fontsize=16,
    )

    if T <= 10:
        tick_step = 1
    elif T <= 15:
        tick_step = 2
    else:
        tick_step = 3
    tick_idx = list(range(0, T, tick_step))
    if tick_idx[-1] != T - 1:
        tick_idx.append(T - 1)

    ax_crps.set_xticks([t_axis[i] for i in tick_idx])
    ax_crps.set_xticklabels([f"{i+1} — {dates[i]}" for i in tick_idx],
                             fontsize=13, rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = args.out or args.npz_path.replace(".npz", "_crps_timeseries.png")
    plt.savefig(out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
