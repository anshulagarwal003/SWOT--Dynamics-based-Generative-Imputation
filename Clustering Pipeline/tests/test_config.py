import pytest

from ssh_clustering import config


def test_get_season_config_builds_expected_path():
    cfg = config.get_season_config("summer", data_root="data/Global")
    assert cfg.data_dir == "data/Global/Global_summer_SSH/"
    assert cfg.name == "summer"


def test_get_season_config_rejects_unknown_season():
    with pytest.raises(ValueError):
        config.get_season_config("autumn", data_root="data/Global")


@pytest.mark.parametrize("season", ["summer", "winter"])
def test_season_cluster_names_length_matches_final_n_clusters(season):
    names = config.SEASON_CLUSTER_NAMES[season]
    assert len(names) == config.FINAL_N_CLUSTERS
