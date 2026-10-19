"""Protocol-validation tests (mode 1). All data here are SYNTHETIC TEST FIXTURES, not results."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/src")
from evaluation import metrics as M, stats as S

SYNTHETIC_FIXTURE = True


def test_mask_iou():
    a = np.zeros((10, 10), bool); a[:5] = True; b = np.zeros((10, 10), bool); b[:, :5] = True
    assert abs(M.mask_iou(a, b) - 25 / 75) < 1e-9


def test_preservation_local_edit():
    x = np.full((32, 32, 3), 100, np.uint8); y = x.copy(); m = np.zeros((32, 32), bool); m[8:16, 8:16] = True; y[m] = 200
    p = M.preservation(x, y, m); assert p["change_out"] == 0 and p["change_in"] == 100


def test_deltae_zero():
    x = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8); assert M.delta_e_in_mask(x, x, np.ones((16, 16), bool)) == 0


def test_bootstrap_detects_shift():
    rng = np.random.default_rng(0); a = rng.normal(1, 1, 200); b = a - 0.5 + rng.normal(0, .1, 200)
    r = S.cluster_bootstrap_diff(a, b, n=2000); assert r["ci95"][0] > 0.4 and r["ci95"][1] < 0.6


def test_cluster_bootstrap_wider_with_clusters():
    rng = np.random.default_rng(1); cl = np.repeat(np.arange(20), 10); eff = rng.normal(0, 1, 20)[cl]
    a = eff + rng.normal(0, .1, 200); b = np.zeros(200)
    w_iid = np.diff(S.cluster_bootstrap_diff(a, b, n=2000)["ci95"])[0]; w_cl = np.diff(S.cluster_bootstrap_diff(a, b, cl, n=2000)["ci95"])[0]
    assert w_cl > 2 * w_iid


def test_mcnemar_and_holm():
    a = np.array([1] * 30 + [0] * 70, bool); b = np.array([1] * 10 + [0] * 90, bool)
    r = S.mcnemar_exact(a, b); assert r["a_only"] == 20 and r["p"] < 1e-4
    h = S.holm({"x": 0.01, "y": 0.04}); assert abs(h["x"] - 0.02) < 1e-12 and abs(h["y"] - 0.04) < 1e-12


def test_macro_f1():
    assert S.macro_f1(np.array([0, 1, 1]), np.array([0, 1, 1])) == 1.0
