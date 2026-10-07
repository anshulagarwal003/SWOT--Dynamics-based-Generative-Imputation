import numpy as np
import xarray as xr
import xrft
import matplotlib.pyplot as plt


def _compute_1d_psd(arr, d, direction):
    min_pts = 10
    all_psds, all_weights, k_ref = [], [], None

    N = arr.shape[1] if direction == 'col' else arr.shape[0]

    for i in range(N):
        line = arr[:, i] if direction == 'col' else arr[i, :]
        n = len(line)
        if n < min_pts:
            continue

        x = np.arange(n) * d
        seg = line - np.polyval(np.polyfit(x, line, 1), x)

        win = np.hanning(n)
        sw = seg * win / np.sqrt(np.mean(win ** 2))

        da = xr.DataArray(sw, dims=('x',), coords={'x': x})
        ps = xrft.power_spectrum(da, dim='x', scaling='density', detrend=None)

        k = ps.freq_x.values
        mask = k > 0
        all_psds.append(ps.values[mask])
        all_weights.append(n)
        if k_ref is None:
            k_ref = k[mask]

    if not all_psds:
        return None, None

    psds_interp = [
        np.interp(k_ref, np.linspace(k_ref[0], k_ref[-1], len(p)), p)
        for p in all_psds
    ]
    return k_ref, np.average(psds_interp, axis=0, weights=all_weights)


def compute_isotropic_psd(arr_2d, dx, dy):
    """
    Compute isotropic 1D PSD from a 2D array with no NaNs.

    Parameters
    ----------
    arr_2d : np.ndarray, shape (Nlon, Nlat)
    dx : float
        Grid spacing in km along longitude.
    dy : float
        Grid spacing in km along latitude.

    Returns
    -------
    k : np.ndarray
        Wavenumber array (cpkm).
    psd : np.ndarray
        Isotropic PSD (m² cpkm⁻¹).
    """
    k, psd_lon = _compute_1d_psd(arr_2d, dx, 'col')
    _, psd_lat = _compute_1d_psd(arr_2d, dy, 'row')
    return k, (psd_lon + psd_lat) / 2


def calculate_and_plot_isotropic_psd(frame_2d, dx, dy):
    """Plot isotropic PSD for a single 2D frame."""
    k, psd = compute_isotropic_psd(frame_2d, dx, dy)
    if k is None:
        print("Could not compute PSD.")
        return
    plt.figure(figsize=(8, 6))
    plt.loglog(k, psd)
    plt.title('Isotropic 1D PSD')
    plt.xlabel('Wavenumber (cpkm)')
    plt.ylabel('PSD (m² cpkm⁻¹)')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     calculate_and_plot_isotropic_psd(data_prediction[0], dx, dy)
