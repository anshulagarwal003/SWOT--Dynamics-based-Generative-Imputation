import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ssh_clustering import config
from ssh_clustering.plotting import grid_labels, plot_cluster_map


def test_grid_labels_shape_matches_grid():
    lon = np.array([0.0, 90.0, -90.0, 180.0])
    lat = np.array([0.0, 30.0, -30.0, 60.0])
    labels = np.array([0, 1, 2, 3])

    loni, lati, gridded = grid_labels(lon, lat, labels, dx=30, lon_range=(-180, 180), lat_range=(-60, 60))

    assert gridded.shape == (len(lati), len(loni))


def test_grid_labels_wraps_longitude_convention():
    # 350 degrees (0-360 convention) should behave like -10 degrees.
    lon_0_360 = np.array([350.0])
    lon_neg180_180 = np.array([-10.0])
    lat = np.array([0.0])
    labels = np.array([1])

    _, _, gridded_a = grid_labels(lon_0_360, lat, labels, dx=10, lon_range=(-180, 180), lat_range=(-10, 10))
    _, _, gridded_b = grid_labels(lon_neg180_180, lat, labels, dx=10, lon_range=(-180, 180), lat_range=(-10, 10))

    np.testing.assert_array_equal(gridded_a, gridded_b)


def test_plot_cluster_map_rejects_mismatched_cluster_names():
    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 10.0])
    labels = np.array([0, 1])
    mask = np.zeros((3, 3))

    with pytest.raises(ValueError):
        plot_cluster_map(
            lon, lat, labels, mask,
            cluster_names=["only one name"],
            dx=10, lon_range=(-10, 10), lat_range=(-10, 10),
        )


def test_plot_cluster_map_with_regime_colors_runs():
    lon = np.array([0.0, 10.0, 20.0, 30.0])
    lat = np.array([0.0, 10.0, -10.0, 20.0])
    labels = np.array([0, 1, 2, 3])
    mask = np.ones((5, 5))  # (lat, lon) grid from dx=10 over [-20, 20]

    cluster_names = config.SEASON_CLUSTER_NAMES["winter"]
    fig = plot_cluster_map(
        lon, lat, labels, mask,
        cluster_names=cluster_names,
        dx=10, lon_range=(-20, 20), lat_range=(-20, 20),
        regime_colors=config.REGIME_COLORS,
    )
    assert fig is not None
