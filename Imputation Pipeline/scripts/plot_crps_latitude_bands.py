"""
Aggregate CRPS results (from compute_global_crps.py) into 10-degree latitude
bands and plot as horizontal stripes over a world map — shows the
equator-to-pole gradient directly instead of scattered individual points.

Usage:
    python plot_crps_latitude_bands.py --csv crps_results_chunk10_winter.csv \
        --seasons winter --out crps_bands_chunk10_winter.png
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BIN_EDGES  = [0.0, 0.2, 0.4, np.inf]
BIN_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]
BIN_LABELS = ["0-0.2", "0.2-0.4", "0.4+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seasons", nargs="+", default=["summer", "winter"])
    ap.add_argument("--band-width", type=int, default=10, help="Latitude band width in degrees (default: 10)")
    ap.add_argument("--out", default="crps_bands.png")
    args = ap.parse_args()
    BAND_WIDTH = args.band_width

    # per (lat_band, season) -> list of crps values
    buckets = defaultdict(list)
    chunk_size = None
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            season = row["season"]
            if season not in args.seasons:
                continue
            lat = float(row["lat"])
            band = int(np.floor(lat / BAND_WIDTH) * BAND_WIDTH)  # e.g. 23.5 -> 20, -5 -> -10
            buckets[(band, season)].append(float(row["crps"]))
            chunk_size = row["chunk_size"]

    cmap = mcolors.ListedColormap(BIN_COLORS)
    norm = mcolors.BoundaryNorm(BIN_EDGES[:-1] + [BIN_EDGES[-2] + 1], cmap.N)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, len(args.seasons), figsize=(9 * len(args.seasons), 6.5),
        subplot_kw={"projection": proj}, squeeze=False,
    )
    axes = axes[0]

    for ax, season in zip(axes, args.seasons):
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=3)
        ax.set_extent([-180, 180, -90, 90], crs=proj)

        n_locations = 0
        for band_start in range(-90, 90, BAND_WIDTH):
            vals = buckets.get((band_start, season))
            if not vals:
                continue
            n_locations += len(vals)
            mean_crps = np.mean(vals)
            color = cmap(norm(mean_crps))
            ax.add_patch(plt.Rectangle(
                (-180, band_start), 360, BAND_WIDTH,
                facecolor=color, edgecolor="black", linewidth=0.3,
                alpha=0.75, zorder=1, transform=proj,
            ))
            ax.text(185, band_start + BAND_WIDTH / 2, f"{mean_crps:.2f}",
                    fontsize=6, va="center", transform=proj)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                           ylocs=range(-90, 91, BAND_WIDTH))
        ax.set_title(f"{season} (n={n_locations} chunks)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    bin_centers = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        bin_centers.append(lo + 0.1 if np.isinf(hi) else (lo + hi) / 2)
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.06, pad=0.1,
                         ticks=bin_centers)
    cbar.ax.set_xticklabels(BIN_LABELS)
    cbar.set_label("mean CRPS")

    plt.suptitle(f"CRPS by {BAND_WIDTH}° latitude band — chunk_size={chunk_size}")
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
