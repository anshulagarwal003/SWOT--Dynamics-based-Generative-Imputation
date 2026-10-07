"""End-to-end pipeline: load spectra, reduce dimensions, cluster, diagnose.

This ties together data_loading / clustering / diagnostics for one season
and is what both the notebooks and the CLI call into.
"""

from dataclasses import dataclass

import pandas as pd

from . import clustering, config, data_loading, diagnostics


@dataclass
class SeasonPipelineResult:
    season: str
    dataset: data_loading.SpectraDataset
    spectra_pca: "object"
    model_selection: clustering.ModelSelectionResult
    clustering_result: clustering.ClusteringResult
    cluster_df: pd.DataFrame
    cluster_diagnostics: list


def run_season_pipeline(
    season: str,
    data_root: str,
    n_components: int = config.FINAL_N_CLUSTERS,
    run_model_selection: bool = True,
) -> SeasonPipelineResult:
    """Run the full pipeline for one season and return every intermediate result.

    Set `run_model_selection=False` to skip the BIC/AIC/silhouette scan over
    `config.GMM_SEARCH_RANGE` (useful once you've already picked
    `n_components` and just want the final clustering + diagnostics).
    """
    season_cfg = config.get_season_config(season, data_root)

    dataset = data_loading.load_season_spectra(
        season_cfg.data_dir, config.WVNUM_CUTOFF_IND, config.FREQ_CUTOFF_IND
    )

    spectra_pca, _pca = clustering.reduce_dimensions(
        dataset.spectra_norm_flat, config.PCA_VARIANCE_TARGET
    )

    model_selection = None
    if run_model_selection:
        model_selection = clustering.select_n_components(
            spectra_pca, config.GMM_SEARCH_RANGE, config.RANDOM_STATE, config.GMM_N_INIT
        )

    clustering_result = clustering.fit_gmm(
        spectra_pca, n_components, config.RANDOM_STATE, config.GMM_N_INIT
    )

    cluster_df = pd.DataFrame(
        {"data_path_ssh": dataset.file_paths, "labels": clustering_result.labels}
    )

    cluster_diagnostics = diagnostics.summarize_all_clusters(
        cluster_df, config.WVNUM_CUTOFF_IND, config.FREQ_CUTOFF_IND
    )

    return SeasonPipelineResult(
        season=season,
        dataset=dataset,
        spectra_pca=spectra_pca,
        model_selection=model_selection,
        clustering_result=clustering_result,
        cluster_df=cluster_df,
        cluster_diagnostics=cluster_diagnostics,
    )
