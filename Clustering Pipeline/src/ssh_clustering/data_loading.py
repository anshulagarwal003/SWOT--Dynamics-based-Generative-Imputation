"""Load per-location wavenumber-frequency SSH spectra from netCDF files.

Each input file is expected to hold four variables for one grid location:
`Wvnumber`, `Frequency`, `Spectrum` (2D, wavenumber x frequency), and scalar
`Latitude` / `Longitude`.
"""

import glob
from dataclasses import dataclass

import netCDF4 as nc
import numpy as np
from sklearn import preprocessing


@dataclass
class SpectraDataset:
    file_paths: list
    lon: np.ndarray
    lat: np.ndarray
    spectra_norm_flat: np.ndarray  # (n_locations, wvnum_cutoff * freq_cutoff), log + L2-normalized


def _read_preprocessed_spectrum(file_path: str, wvnum_cutoff: int, freq_cutoff: int) -> np.ndarray:
    """Read one file and return its wvnum x freq spectrum, scaled by wvnum * freq.

    Returns array of shape (freq_cutoff, wvnum_cutoff), matching the
    transpose convention used throughout the original analysis.
    """
    ds = nc.Dataset(file_path)
    wvnum = ds["Wvnumber"][:wvnum_cutoff]
    freq = ds["Frequency"][:freq_cutoff]
    spectrum = ds["Spectrum"][:wvnum_cutoff, :freq_cutoff]

    freq_mat = np.tile(freq, (wvnum.size, 1))
    wvnum_mat = np.transpose(np.tile(wvnum, (freq.size, 1)))
    return np.transpose(spectrum * wvnum_mat * freq_mat)


def load_season_spectra(data_dir: str, wvnum_cutoff: int, freq_cutoff: int) -> SpectraDataset:
    """Load, preprocess and flatten all spectra for one season.

    Preprocessing per file: multiply by wavenumber and frequency (to
    compensate for the spectral slope), take the log, and L2-normalize the
    flattened spectrum. This mirrors the original notebooks' `spectra_norm`
    step ahead of PCA.
    """
    file_paths = sorted(glob.glob(f"{data_dir}*.nc"))
    if not file_paths:
        raise FileNotFoundError(f"No .nc files found under {data_dir}")

    n_files = len(file_paths)
    spectra_norm_flat = np.zeros((n_files, wvnum_cutoff * freq_cutoff))
    lat_list, lon_list = [], []

    for i, file_path in enumerate(file_paths):
        raw = _read_preprocessed_spectrum(file_path, wvnum_cutoff, freq_cutoff)
        spectra_norm_flat[i, :] = preprocessing.normalize(np.log(raw)).flatten()

        ds = nc.Dataset(file_path)
        lat_list.append(np.atleast_1d(ds.variables["Latitude"][:]))
        lon_list.append(np.atleast_1d(ds.variables["Longitude"][:]))

    lat = np.concatenate(lat_list).flatten()
    lon = np.concatenate(lon_list).flatten()

    return SpectraDataset(
        file_paths=file_paths,
        lon=lon,
        lat=lat,
        spectra_norm_flat=spectra_norm_flat,
    )


def read_raw_spectrum(file_path: str, wvnum_cutoff: int, freq_cutoff: int):
    """Return (wvnum, freq, preprocessed_spectrum) for one file, for plotting."""
    ds = nc.Dataset(file_path)
    wvnum = ds["Wvnumber"][:wvnum_cutoff]
    freq = ds["Frequency"][:freq_cutoff]
    spectrum = _read_preprocessed_spectrum(file_path, wvnum_cutoff, freq_cutoff)
    return wvnum, freq, spectrum
