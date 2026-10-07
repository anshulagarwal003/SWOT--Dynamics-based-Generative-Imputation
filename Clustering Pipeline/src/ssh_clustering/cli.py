"""Command-line entry point: run the pipeline for one season and save figures.

Example:
    python -m ssh_clustering.cli --season winter --data-root data/Global \\
        --mask data/mask_to_plot_ratio.npz --out-dir figures/winter
"""

import argparse
import os

from . import config, diagnostics, masks, plotting
from .pipeline import run_season_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, choices=["summer", "winter"])
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing Global_<season>_SSH/ subfolders of .nc files",
    )
    parser.add_argument("--mask", required=True, help="Path to the land mask .npz file")
    parser.add_argument("--out-dir", required=True, help="Directory to write figures into")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=config.FINAL_N_CLUSTERS,
        help=f"Number of GMM clusters to fit (default: {config.FINAL_N_CLUSTERS}, the published choice)",
    )
    parser.add_argument(
        "--skip-model-selection",
        action="store_true",
        help="Skip the BIC/AIC/silhouette scan and only fit the final model",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    result = run_season_pipeline(
        season=args.season,
        data_root=args.data_root,
        n_components=args.n_clusters,
        run_model_selection=not args.skip_model_selection,
    )

    if result.model_selection is not None:
        fig = plotting.plot_model_selection(result.model_selection, title_suffix=f"({args.season})")
        fig.savefig(os.path.join(args.out_dir, "model_selection.png"), dpi=150)

    fig = plotting.plot_pca_scatter(
        result.spectra_pca, result.clustering_result.labels, title=f"GMM clusters ({args.season})"
    )
    fig.savefig(os.path.join(args.out_dir, "pca_scatter.png"), dpi=150)

    cluster_names = config.SEASON_CLUSTER_NAMES.get(args.season)
    if cluster_names and len(cluster_names) == args.n_clusters:
        land_mask = masks.load_land_mask(args.mask)
        fig = plotting.plot_cluster_map(
            lon=result.dataset.lon,
            lat=result.dataset.lat,
            labels=result.clustering_result.labels,
            mask=land_mask,
            cluster_names=cluster_names,
            dx=config.GRID_DX,
            lon_range=config.GRID_LON_RANGE,
            lat_range=config.GRID_LAT_RANGE,
            title=f"Global cluster distribution — {args.season}",
            regime_colors=config.REGIME_COLORS,
        )
        fig.savefig(os.path.join(args.out_dir, "cluster_map.png"), dpi=150)
    else:
        print(
            f"Skipping cluster map: no configured names for {args.n_clusters} clusters "
            f"in season '{args.season}' (edit config.SEASON_CLUSTER_NAMES)."
        )

    percentages = diagnostics.cluster_size_percentages(result.clustering_result.labels)
    print("Cluster size (% of locations):")
    print(percentages.to_string())

    for diag in result.cluster_diagnostics:
        fig = plotting.plot_mean_spectrum(diag, title=f"{args.season} — cluster {diag.cluster_label}")
        fig.savefig(os.path.join(args.out_dir, f"mean_spectrum_cluster{diag.cluster_label}.png"), dpi=150)
        print(f"Cluster {diag.cluster_label}: mean RMS = {diag.mean_rms:.2f}")

    print(f"Done. Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
