"""Plotting helpers: model-selection curves, spectra, and global cluster maps."""

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from scipy.interpolate import griddata


def plot_model_selection(result, title_suffix: str = ""):
    """Plot BIC, AIC, and silhouette score vs. number of GMM components."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    n = list(result.n_components_range)

    axes[0].plot(n, result.bic_scores, marker="o")
    axes[0].set(xlabel="Number of components", ylabel="BIC", title=f"BIC {title_suffix}".strip())

    axes[1].plot(n, result.aic_scores, marker="o", color="tab:orange")
    axes[1].set(xlabel="Number of components", ylabel="AIC", title=f"AIC {title_suffix}".strip())

    axes[2].plot(n, result.silhouette_scores, marker="o", color="tab:green")
    axes[2].set(
        xlabel="Number of components",
        ylabel="Silhouette score",
        title=f"Silhouette {title_suffix}".strip(),
    )

    fig.tight_layout()
    return fig


def plot_pca_scatter(spectra_pca: np.ndarray, labels: np.ndarray, title: str = "GMM clusters"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(spectra_pca[:, 0], spectra_pca[:, 1], c=labels, cmap="viridis", marker="o")
    ax.set(xlabel="PCA component 1", ylabel="PCA component 2", title=title)
    fig.colorbar(sc, ax=ax, label="Cluster")
    return fig


def plot_mean_spectrum(diagnostics, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 5))
    pcm = ax.pcolormesh(
        diagnostics.wvnum,
        diagnostics.freq,
        diagnostics.mean_spectrum,
        cmap="inferno",
        norm=mcolors.LogNorm(vmin=1e-6, vmax=1e-3),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title or f"Mean spectrum — cluster {diagnostics.cluster_label}")
    fig.colorbar(pcm, ax=ax)
    return fig


def grid_labels(lon: np.ndarray, lat: np.ndarray, labels: np.ndarray, dx: int, lon_range, lat_range):
    """Interpolate per-location cluster labels onto a regular lat/lon grid.

    Longitude is normalized to [-180, 180) before gridding so inputs given
    in either 0-360 or -180-180 convention are handled the same way.
    """
    lon_wrapped = ((lon + 180) % 360) - 180
    loni = np.arange(lon_range[0], lon_range[1] + dx, dx)
    lati = np.arange(lat_range[0], lat_range[1] + dx, dx)
    long, latg = np.meshgrid(loni, lati)
    labels_gridded = griddata(np.stack([lon_wrapped, lat], axis=1), labels, (long, latg), method="nearest")
    return loni, lati, labels_gridded


def plot_cluster_map(
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    cluster_names: list,
    dx: int,
    lon_range,
    lat_range,
    title: str = "Global cluster distribution",
    regime_colors: dict | None = None,
):
    """Plot a global map of dominant cluster per grid cell, land masked out.

    `mask` should be a 2D array matching the (lat, lon) grid shape produced
    from `dx`/`lon_range`/`lat_range`, nonzero over land.

    If `regime_colors` is given (a dict mapping entries of `cluster_names` to
    hex colors), each regime gets that fixed color regardless of its cluster
    index — this is what keeps summer and winter maps color-consistent for
    the same physical regime. Otherwise falls back to an evenly spaced Set3
    palette keyed by cluster index.
    """
    loni, lati, labels_gridded = grid_labels(lon, lat, labels, dx, lon_range, lat_range)

    n_clusters = len(cluster_names)
    if n_clusters != len(np.unique(labels)):
        raise ValueError(
            f"cluster_names has {n_clusters} entries but data has {len(np.unique(labels))} clusters"
        )

    labels_gridded = labels_gridded.astype(float) * mask
    mask_plot = mask.copy()
    mask_plot[mask_plot > 0] = np.nan

    if regime_colors is not None:
        cmap = mcolors.ListedColormap([regime_colors[name] for name in cluster_names])
    else:
        cmap = plt.get_cmap("Set3", n_clusters)
    bounds = np.arange(-0.5, n_clusters, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(12, 5), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolor(loni, lati, labels_gridded, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, ticks=range(n_clusters))
    cbar.ax.set_yticklabels(cluster_names)

    ax.pcolor(loni, lati, mask_plot, cmap="Greys", vmin=-1, vmax=1, transform=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.7, color="black")

    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 61, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_extent([*lon_range, *lat_range], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    fig.tight_layout()
    return fig
