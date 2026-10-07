"""PCA dimensionality reduction and GMM clustering of SSH spectra."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def reduce_dimensions(spectra_norm_flat: np.ndarray, variance_target: float) -> tuple[np.ndarray, PCA]:
    """Fit PCA to reach `variance_target` explained variance and transform the data."""
    pca = PCA(variance_target)
    pca.fit(spectra_norm_flat)
    spectra_pca = pca.transform(spectra_norm_flat)
    return spectra_pca, pca


@dataclass
class ModelSelectionResult:
    n_components_range: range
    bic_scores: list
    aic_scores: list
    silhouette_scores: list

    @property
    def optimal_n_bic(self) -> int:
        return self.n_components_range[int(np.argmin(self.bic_scores))]

    @property
    def optimal_n_aic(self) -> int:
        return self.n_components_range[int(np.argmin(self.aic_scores))]

    @property
    def optimal_n_silhouette(self) -> int:
        return self.n_components_range[int(np.argmax(self.silhouette_scores))]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "n_components": list(self.n_components_range),
                "bic": self.bic_scores,
                "aic": self.aic_scores,
                "silhouette": self.silhouette_scores,
            }
        )


def select_n_components(
    spectra_pca: np.ndarray,
    n_components_range: range,
    random_state: int,
    n_init: int,
) -> ModelSelectionResult:
    """Score candidate GMM cluster counts by BIC, AIC, and silhouette score."""
    bic_scores, aic_scores, silhouette_scores = [], [], []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=n_init)
        gmm.fit(spectra_pca)
        labels = gmm.predict(spectra_pca)

        bic_scores.append(gmm.bic(spectra_pca))
        aic_scores.append(gmm.aic(spectra_pca))
        silhouette_scores.append(silhouette_score(spectra_pca, labels))

    return ModelSelectionResult(
        n_components_range=n_components_range,
        bic_scores=bic_scores,
        aic_scores=aic_scores,
        silhouette_scores=silhouette_scores,
    )


@dataclass
class ClusteringResult:
    gmm: GaussianMixture
    labels: np.ndarray
    probabilities: np.ndarray


def fit_gmm(spectra_pca: np.ndarray, n_components: int, random_state: int, n_init: int) -> ClusteringResult:
    """Fit the final GMM with a fixed number of components and a fixed seed for reproducibility."""
    np.random.seed(random_state)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=n_init)
    gmm.fit(spectra_pca)
    labels = gmm.predict(spectra_pca)
    probabilities = gmm.predict_proba(spectra_pca)
    return ClusteringResult(gmm=gmm, labels=labels, probabilities=probabilities)
