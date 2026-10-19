"""Tests for src/review/agreement.py. All rating data here are SIMULATED or textbook examples;
no real expert ratings exist yet."""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/src")
from review import agreement as ag  # noqa: E402

nan = np.nan
# Krippendorff (2011) "Computing Krippendorff's alpha-reliability", example C: 4 observers x 12 units, missing data.
K2011 = np.array([
    [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
    [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan]])


@pytest.mark.parametrize("level,expected", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849),
                                            ("ratio", 0.797)])
def test_krippendorff_2011_worked_example(level, expected):
    assert ag.krippendorff_alpha(K2011, level) == pytest.approx(expected, abs=5e-4)


def test_krippendorff_2011_binary_example():
    # Krippendorff (2011) example A: two observers, binary data, alpha = 0.095
    d = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]], float)
    assert ag.krippendorff_alpha(d, "nominal") == pytest.approx(0.095, abs=5e-4)


def test_alpha_perfect_agreement_is_one():
    x = np.array([1, 2, 3, 4, 5, 3, 2, 4], float)
    assert ag.krippendorff_alpha(np.vstack([x, x, x]), "ordinal") == pytest.approx(1.0)


def test_alpha_random_ratings_near_zero():
    rng = np.random.default_rng(0)
    d = rng.integers(1, 6, size=(3, 3000)).astype(float)
    assert abs(ag.krippendorff_alpha(d, "ordinal")) < 0.03


def test_alpha_bootstrap_ci_contains_estimate():
    est, lo, hi = ag.alpha_bootstrap(K2011, "ordinal", reps=300, seed=1)
    assert lo <= est <= hi


def test_weighted_kappa_perfect_and_reversed():
    x = np.array([1, 2, 3, 4, 5] * 4, float)
    assert ag.weighted_kappa(x, x) == pytest.approx(1.0)
    assert ag.weighted_kappa(x, 6 - x) == pytest.approx(-1.0)


def test_weighted_kappa_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    x = rng.integers(1, 6, 200)
    y = np.clip(x + rng.integers(-1, 2, 200), 1, 5)
    ref = skm.cohen_kappa_score(x, y, weights="quadratic", labels=[1, 2, 3, 4, 5])
    assert ag.weighted_kappa(x.astype(float), y.astype(float)) == pytest.approx(ref, abs=1e-12)


def test_weighted_kappa_random_near_zero():
    rng = np.random.default_rng(4)
    x, y = rng.integers(1, 6, 5000).astype(float), rng.integers(1, 6, 5000).astype(float)
    assert abs(ag.weighted_kappa(x, y)) < 0.05


def test_icc_shrout_fleiss_1979_table():
    # Shrout & Fleiss (1979) Table 2: 6 targets x 4 judges; Table 4 reports the ICCs to two decimals.
    Y = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    r = ag.icc(Y)
    for k, v in {"ICC(1,1)": .17, "ICC(2,1)": .29, "ICC(3,1)": .71, "ICC(1,k)": .44, "ICC(2,k)": .62,
                 "ICC(3,k)": .91}.items():
        assert r[k] == pytest.approx(v, abs=0.006), k


def test_holm_and_signflip():
    assert ag.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ag.signflip_p(np.ones(6)) == pytest.approx(2 / 64)


def test_end_to_end_on_simulated_package(tmp_path):
    """SIMULATED ratings on a synthetic key: agreement must be positive, catch items must be passed,
    and the built-in method effect (KB-3D > generic-3D) must be recovered in direction."""
    key = tmp_path / "key.csv"
    rng = np.random.default_rng(0)
    rows, page = [], 1
    fams = {"H": "KB-3D", "E": "generic-3D", "C_stage": "KB-2D", "B": "generic-2D"}
    for s in range(9):
        for m, fam in fams.items():
            rows.append(dict(item_id=f"X-{page:04d}", page=page, item_key=f"S{s}|{m}", specimen=f"S{s}", kb_key="k",
                             method=m, method_family=fam, provenance="SYNTHETIC", kind="3d", role="test",
                             is_duplicate=0, duplicate_of_item_id=""))
            page += 1
    for i in range(5):
        r = dict(rows[i * 3], item_id=f"X-D{i:03d}", page=page, is_duplicate=1,
                 duplicate_of_item_id=rows[i * 3]["item_id"])
        rows.append(r)
        page += 1
    for i in range(3):
        rows.append(dict(item_id=f"X-C{i:03d}", page=page, item_key=f"S{i}|C_mismatch", specimen=f"S{i}",
                         kb_key="k", method="C_mismatch", method_family="catch", provenance="EDITED-REAL-2D",
                         kind="2d", role="catch", is_duplicate=0, duplicate_of_item_id=""))
        page += 1
    with open(key, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rdir = tmp_path / "ratings"
    ag.simulate(str(key), str(rdir), n_raters=3, seed=2, noise=0.4,
                method_effects={"H": 1.0, "E": -1.0, "C_stage": 0.0, "B": -0.5})
    res = ag.analyse(str(key), [str(rdir)], str(tmp_path / "out"), reps=200)
    assert "SIMULATED" in res["data_status"] and "No real expert ratings" in res["data_status"]
    assert res["agreement"]["per_criterion"]["f_overall_accept"]["alpha"] > 0.3
    assert all(d["pass_rate"] == 1.0 for d in res["agreement"]["catch"].values())
    comp = [c for c in res["method_effect"]["comparisons"]
            if c["criterion"] == "f_overall_accept" and (c["a"], c["b"]) == ("KB-3D", "generic-3D")][0]
    assert comp["mean_diff"] > 0 and comp["ci95"][0] > 0
    for fn in ("agreement_report.md", "per_method_scores.csv", "pairwise_kappa.csv"):
        assert "No real expert ratings" in open(os.path.join(tmp_path / "out", fn)).read()
