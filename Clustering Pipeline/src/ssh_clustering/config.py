"""Configuration for the SSH spectral clustering pipeline.

Edit `FINAL_N_CLUSTERS` and the per-season entries in `SEASON_CLUSTER_NAMES`
if the published number of regimes or their descriptive names ever change —
everything else in the pipeline reads from here rather than hardcoding these
choices inline.
"""

from dataclasses import dataclass, field


# Spectral cutoffs applied when reading each per-location wavenumber-frequency
# spectrum (matches the resolution of the input .nc files).
WVNUM_CUTOFF_IND = 80
FREQ_CUTOFF_IND = 300

# Number of leading principal components is chosen automatically to reach
# this fraction of explained variance.
PCA_VARIANCE_TARGET = 0.95

# Range of candidate cluster counts scanned when selecting the GMM order via
# BIC / AIC / silhouette score.
GMM_SEARCH_RANGE = range(2, 10)

# Final published number of clusters. Change this (and the matching entry in
# SEASON_CLUSTER_NAMES below) to reproduce the pipeline with a different
# number of regimes.
FINAL_N_CLUSTERS = 4

RANDOM_STATE = 42
GMM_N_INIT = 10

# Descriptive regime names for the final clustering, keyed by season and
# then by cluster index (0-based, in the order GaussianMixture assigns
# them). These are free-text scientific labels chosen by inspecting the
# average spectrum of each cluster (see notebooks/) and are only valid for
# FINAL_N_CLUSTERS clusters -- update both together.
SEASON_CLUSTER_NAMES = {
    "summer": [
        "Moderate MBM, IGW",
        "Strong MBM, SBM, USM",
        "Strong IGW",
        "Strong MBM, Weak SBM",
    ],
    "winter": [
        "Strong IGW",
        "Moderate IGW, Weak MBM",
        "Strong MBM, SBM, USM",
        "Strong MBM, Weak SBM",
    ],
}

# Fixed color per physical regime (keyed by name, not by cluster index), so
# the same regime gets the same color in both seasons' maps regardless of
# which index GaussianMixture happens to assign it.
REGIME_COLORS = {
    "Strong IGW": "#80b1d3",
    "Moderate MBM, IGW": "#8dd3c7",
    "Moderate IGW, Weak MBM": "#8dd3c7",
    "Strong MBM, SBM, USM": "#fb8072",
    "Strong MBM, Weak SBM": "#fdb462",
}

# Grid spacing (degrees) used when interpolating per-location cluster labels
# onto a regular lat/lon grid for map plots.
GRID_DX = 6
GRID_LON_RANGE = (-180, 180)
GRID_LAT_RANGE = (-70, 70)


@dataclass(frozen=True)
class SeasonConfig:
    """Per-season paths and metadata."""

    name: str
    data_dir: str
    cluster_names: list = field(default_factory=list)


def get_season_config(season: str, data_root: str) -> SeasonConfig:
    if season not in ("summer", "winter"):
        raise ValueError(f"Unknown season '{season}', expected 'summer' or 'winter'")
    return SeasonConfig(
        name=season,
        data_dir=f"{data_root.rstrip('/')}/Global_{season}_SSH/",
        cluster_names=SEASON_CLUSTER_NAMES.get(season, []),
    )
