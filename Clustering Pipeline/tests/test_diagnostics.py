import numpy as np
import pytest

from ssh_clustering.diagnostics import calculate_rms, cluster_size_percentages


def test_calculate_rms_zero_for_identical_matrices():
    m = np.ones((5, 5))
    assert calculate_rms(m, m) == 0.0


def test_calculate_rms_matches_manual_computation():
    m1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    m2 = np.array([[1.0, 1.0], [1.0, 1.0]])
    expected = 1_000_000 * np.sqrt(np.mean((m1 - m2) ** 2))
    assert calculate_rms(m1, m2) == pytest.approx(expected)


def test_calculate_rms_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        calculate_rms(np.zeros((2, 2)), np.zeros((3, 3)))


def test_cluster_size_percentages_sums_to_100():
    labels = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    pct = cluster_size_percentages(labels)
    assert pct.sum() == pytest.approx(100.0)
    assert list(pct.index) == [0, 1, 2, 3]
    assert pct.loc[3] == pytest.approx(40.0)


def test_cluster_size_percentages_matches_known_counts():
    # regression check against the published summer cluster counts
    labels = np.array([0] * 295 + [1] * 181 + [2] * 159 + [3] * 273)
    pct = cluster_size_percentages(labels)
    assert pct.loc[0] == pytest.approx(32.49, abs=0.01)
    assert pct.loc[1] == pytest.approx(19.93, abs=0.01)
    assert pct.loc[2] == pytest.approx(17.51, abs=0.01)
    assert pct.loc[3] == pytest.approx(30.07, abs=0.01)
