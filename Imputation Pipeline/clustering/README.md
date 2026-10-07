# Clustering

The Gaussian Mixture Model (GMM) spectral clustering pipeline used to
identify dynamical regimes (Figures 18–22: BIC-based model selection,
cluster mean spectra, and global spatial distribution of clusters) is
maintained in a separate companion repository:

`<CLUSTERING_REPO_LINK>`

The cluster assignments and centroid coordinates produced by that pipeline
are included here under `../data/regime_labels/`, and are what
`src/compute_global_crps.py` and the regime-conditional CRPS figures
(29–31) in this repository consume.
