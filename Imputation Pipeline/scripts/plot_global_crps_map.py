"""
Same as plot_global_crps_relabeled.py, but with IQR-based outlier removal:
locations whose mean CRPS falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (computed
per chunk_size, combining both seasons -- the same population used for vmax)
are dropped from the plot and grayed out, same as "too far" cells.

Usage:
    python plot_global_crps_relabeled_no_outliers.py --chunk-sizes 5 10 21 --lat-cutoff 60
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from compute_global_crps import index_to_season


def load_and_relabel(csv_path):
    """Read a CRPS results CSV and recompute season correctly from lat + start.
    Does not modify the file on disk."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["lat"])
            start = int(row["start"])
            chunk_size = int(row["chunk_size"])
            row["lat"] = lat
            row["lon"] = float(row["lon"])
            row["crps"] = float(row["crps"])
            row["season"] = index_to_season(start, chunk_size, lat)  # corrected, in-memory only
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[5, 10, 21])
    ap.add_argument("--lat-cutoff", type=float, default=60)
    ap.add_argument("--dx", type=float, default=6, help="Grid spacing in degrees")
    ap.add_argument("--dist-threshold", type=float, default=6.0,
                     help="Grid cells farther than this (degrees) from any real data point are grayed out")
    ap.add_argument("--outlier-k", type=float, default=1.5,
                     help="IQR multiplier for outlier removal (standard boxplot rule uses 1.5)")
    ap.add_argument("--mask-file", default="mask_to_plot_ratio.npz")
    ap.add_argument("--out-prefix", default="crps_map_relabeled_no_outliers")
    args = ap.parse_args()

    dx = args.dx
    LAT_CUTOFF = args.lat_cutoff

    loni = np.arange(-180, 180 + dx, dx)
    lati = np.arange(-70, 70 + dx, dx)
    long, latg = np.meshgrid(loni, lati)
    grid_points = np.column_stack([long.ravel(), latg.ravel()])

    mask_data = np.load(args.mask_file)
    mask = mask_data["mask"][:, :]
    beyond_cutoff = np.abs(lati) > LAT_CUTOFF

    gray_cmap = ListedColormap(["#b0b0b0"])

    combos = [(cs, season) for cs in args.chunk_sizes for season in ("winter", "summer")]

    def get_buckets(chunk_size, season):
        # merge rows from both stale-labeled files for this chunk_size, then filter
        # to the CORRECT season using the relabeled value, since some rows flip
        merged_rows = (
            load_and_relabel(f"crps_results_chunk{chunk_size}_winter.csv")
            + load_and_relabel(f"crps_results_chunk{chunk_size}_summer.csv")
        )

        buckets = defaultdict(list)
        for row in merged_rows:
            if row["season"] != season:
                continue
            lat, lon = row["lat"], row["lon"]
            if abs(lat) > LAT_CUTOFF:
                continue
            buckets[(lat, lon)].append(row["crps"])
        return buckets

    # IQR outlier bounds + shared vmax per chunk_size: computed from the combined
    # winter+summer population for that chunk_size, so removal and color scale
    # stay consistent across the two seasons of the same chunk_size
    bounds_by_chunk_size = {}
    for chunk_size in args.chunk_sizes:
        combined_vals = []
        for season in ("winter", "summer"):
            buckets = get_buckets(chunk_size, season)
            combined_vals.extend(np.mean(v) for v in buckets.values())
        combined_vals = np.array(combined_vals)
        q1, q3 = np.percentile(combined_vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - args.outlier_k * iqr
        upper = q3 + args.outlier_k * iqr
        bounds_by_chunk_size[chunk_size] = (lower, upper)

    panel_data = {}  # (chunk_size, season) -> dict of arrays, saved to .npz after plotting

    def plot_panel(ax, chunk_size, season):
        buckets = get_buckets(chunk_size, season)
        lower, upper = bounds_by_chunk_size[chunk_size]

        lats, lons, vals = [], [], []
        n_outliers = 0
        for (lat, lon), crps_list in buckets.items():
            mean_crps = np.mean(crps_list)
            if mean_crps < lower or mean_crps > upper:
                n_outliers += 1
                continue
            lats.append(lat)
            lons.append(lon)
            vals.append(mean_crps)

        coordinate = np.column_stack([lons, lats])
        crps_vals = np.array(vals)

        crps_gridded = griddata(coordinate, crps_vals, (long, latg), method="nearest")

        tree = cKDTree(coordinate)
        dist_to_nearest, _ = tree.query(grid_points)
        too_far = dist_to_nearest.reshape(long.shape) > args.dist_threshold
        crps_gridded[too_far] = np.nan
        n_too_far = too_far.sum()

        crps_gridded = crps_gridded * mask
        crps_gridded[beyond_cutoff, :] = np.nan

        grayed_out = np.full_like(mask, np.nan)
        grayed_out[mask == 0] = 1.0
        grayed_out[beyond_cutoff, :] = 1.0
        grayed_out[too_far] = 1.0

        vmax = upper

        mesh = ax.pcolor(loni, lati, crps_gridded, cmap="Blues", vmin=0, vmax=vmax,
                          transform=ccrs.PlateCarree())
        plt.colorbar(mesh, ax=ax, shrink=0.7, label="mean CRPS")

        ax.pcolor(loni, lati, grayed_out, cmap=gray_cmap, vmin=0, vmax=1,
                  transform=ccrs.PlateCarree())

        ax.coastlines(resolution="50m", linewidth=0.7, color="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)

        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-60, 61, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

        ax.set_title(f"chunk_size={chunk_size}, {season} (hemisphere-corrected) "
                     f"(n={len(crps_vals)} locations, |lat|<={LAT_CUTOFF}, "
                     f"{n_outliers} outliers dropped, {n_too_far} cells blanked as too far)",
                     fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree())
        ax.set_aspect("auto")

    fig, axes = plt.subplots(len(args.chunk_sizes), 2, figsize=(20, 5 * len(args.chunk_sizes)),
                              subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)

    for row_idx, chunk_size in enumerate(args.chunk_sizes):
        for col_idx, season in enumerate(("winter", "summer")):
            plot_panel(axes[row_idx, col_idx], chunk_size, season)

    plt.suptitle("Global CRPS maps, hemisphere-aware seasons, IQR outliers removed", fontsize=14)
    plt.tight_layout()
    out_path = f"{args.out_prefix}_all6.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
