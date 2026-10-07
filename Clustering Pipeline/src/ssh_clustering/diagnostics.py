"""Per-cluster average spectra and RMS-spread diagnostics.

For each cluster, we average the (wavenumber x frequency) spectra of its
member locations, then measure how far each member's spectrum deviates from
that cluster average (RMS). This is a sanity check on cluster cohesion, not
part of the clustering itself.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loading import read_raw_spectrum


def cluster_size_percentages(labels: np.ndarray) -> pd.Series:
    """Percentage of locations assigned to each cluster, indexed by cluster label.

    This is a point-count percentage (fraction of input locations), not an
    area-weighted percentage — it will generally disagree with how much map
    area a cluster appears to cover, since the map is built by nearest-
    neighbor interpolation onto an unweighted lat/lon grid.
    """
    counts = pd.Series(labels).value_counts().sort_index()
    return (counts / counts.sum() * 100).round(2)


def calculate_rms(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Scaled RMS difference between two equally-shaped spectra."""
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same dimensions")
    return 1_000_000 * np.sqrt(np.mean(np.square(matrix1 - matrix2)))


@dataclass
class ClusterDiagnostics:
    cluster_label: int
    mean_spectrum: np.ndarray
    wvnum: np.ndarray
    freq: np.ndarray
    member_rms: list
    mean_rms: float


def cluster_average_and_rms(
    file_paths: list,
    cluster_label: int,
    wvnum_cutoff: int,
    freq_cutoff: int,
) -> ClusterDiagnostics:
    """Compute the mean spectrum for one cluster and each member's RMS deviation from it."""
    mean_spectrum = np.zeros((freq_cutoff, wvnum_cutoff))
    wvnum = freq = None

    spectra = []
    for path in file_paths:
        wvnum, freq, spectrum = read_raw_spectrum(path, wvnum_cutoff, freq_cutoff)
        spectra.append(spectrum)
        mean_spectrum += spectrum
    mean_spectrum /= len(file_paths)

    member_rms = [calculate_rms(mean_spectrum, spectrum) for spectrum in spectra]

    return ClusterDiagnostics(
        cluster_label=cluster_label,
        mean_spectrum=mean_spectrum,
        wvnum=wvnum,
        freq=freq,
        member_rms=member_rms,
        mean_rms=float(np.mean(member_rms)),
    )


def summarize_all_clusters(
    df: pd.DataFrame,
    wvnum_cutoff: int,
    freq_cutoff: int,
) -> list:
    """Run `cluster_average_and_rms` for every cluster label present in df.

    `df` must have columns 'data_path_ssh' and 'labels' (see clustering.fit_gmm
    output joined with the source file paths).
    """
    results = []
    for cluster_label, group in df.groupby("labels"):
        results.append(
            cluster_average_and_rms(
                file_paths=group["data_path_ssh"].tolist(),
                cluster_label=cluster_label,
                wvnum_cutoff=wvnum_cutoff,
                freq_cutoff=freq_cutoff,
            )
        )
    return results
