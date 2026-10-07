# Data

This repository does not ship the raw SSH (sea surface height) spectral data.

## Expected layout

Place the input netCDF files under:

```
data/Global/Global_summer_SSH/*.nc
data/Global/Global_winter_SSH/*.nc
```

Each `.nc` file corresponds to one grid location and must contain:

| Variable    | Shape                       | Description                              |
|-------------|------------------------------|-------------------------------------------|
| `Wvnumber`  | (n_wavenumber,)              | Wavenumber axis                           |
| `Frequency` | (n_frequency,)                | Frequency axis                            |
| `Spectrum`  | (n_wavenumber, n_frequency)   | Wavenumber-frequency power spectrum       |
| `Latitude`  | scalar                        | Location latitude                         |
| `Longitude` | scalar                        | Location longitude                        |

Only the first `WVNUM_CUTOFF_IND` (80) wavenumber bins and `FREQ_CUTOFF_IND`
(300) frequency bins are used — see [`src/ssh_clustering/config.py`](../src/ssh_clustering/config.py).

## Source

SSH is taken from the **MITgcm-LLC4320** global high-resolution (~1/48°)
ocean simulation, used as a ground-truth proxy in place of satellite
altimetry (see the parent thesis for why: altimetry's revisit time is too
coarse to resolve the fast-evolving features this analysis targets).
Wavenumber-frequency spectra were computed per grid location from LLC4320
SSH output for each season (summer / winter); [fill in the exact time range,
spatial sampling, and spectral-estimation method used — e.g. windowing,
detrending, segment length — before publishing, so results are
reproducible from the raw model output].

## Land/ocean mask

`mask_to_plot_ratio.npz` (included in this repo) is a small (25x61) grid mask
used only to blank out land in the global cluster-distribution map plots. It
contains a single array `mask`, nonzero over land, on the same regular
lat/lon grid produced by `config.GRID_DX`/`GRID_LON_RANGE`/`GRID_LAT_RANGE`.
