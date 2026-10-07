# Data

## What's included in this repository

- `regime_labels/` — GMM cluster assignments and centroid coordinates for each
  (season, number-of-clusters) configuration used in the regime-conditional
  CRPS analysis, plus the coordinate lookup table (`file_coordinates.csv`)
  mapping each raw SSH time series file to its (latitude, longitude).
- `derived_crps/` — per-chunk, per-location CRPS results on the held-out test
  set (`crps_results_chunk{5,10,21}_{summer,winter}.csv`), the cluster-mean
  CRPS summaries used in Figures 29–31, the best/worst-CRPS case lookup table
  (`extreme_cases_lookup.csv`), and the land/distance mask used for the global
  CRPS maps (`mask_to_plot_ratio.npz`).
- `derived_npz/` — precomputed reconstruction outputs (ground truth, linear
  interpolation, U-Net prediction, diffusion ensemble samples, isotropic PSD
  curves, and per-pixel CRPS maps) for the representative location and gap
  lengths shown in Figures 23–28, and for the twelve best/worst-CRPS test
  cases. These let `scripts/compare_models_plot.py` regenerate every
  reconstruction figure in the paper without rerunning model inference.

## What's not included

- **Raw SSH time series** (`.npy` files, one per spatial location, ~27 GB
  total) — too large for this repository. Available at: `<DATA_DOWNLOAD_LINK>`.
  Extract to a local directory and point `src/config.yaml`'s `data.folder`
  at it (or set the `--data-folder` argument on the relevant scripts).
- **Full training checkpoints / W&B run logs** — only the selected best
  checkpoint per model/chunk-size is archived (see `../checkpoints/`); every
  intermediate checkpoint from hyperparameter sweeps is not included here.
- **GMM clustering and BIC-selection code/outputs** (Figures 18–22) — this
  lives in a companion repository: `<CLUSTERING_REPO_LINK>`.

## Regenerating `derived_npz/` and `derived_crps/` from raw data

1. Download and extract the raw SSH data (see above) and set `data.folder`
   in `src/config.yaml`.
2. Train or download the checkpoints (see `../checkpoints/README.md`).
3. Run `scripts/compare_models.py` with the desired `--lat --lon --start
   --chunk-size` to produce a `compare_*.npz` (saved under `derived_npz/`
   by convention).
4. Run `src/compute_global_crps.py` to produce the
   `crps_results_chunk*_{summer,winter}.csv` files in `derived_crps/`.
