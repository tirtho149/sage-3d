"""Unit tests for the v2 operators on synthetic geometry with known answers (no SAM 3D asset, no GPU)."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from geometry_editor.curl import parse, curl_target
from geometry_editor import bilayer, gall
from metrics.structure3d import pair_strain


def flat_sheet(nu=120, nv=60, w=0.5):
    u, v = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(-w, w, nv), indexing="ij")
    X = np.stack([u.ravel(), v.ravel(), np.zeros(u.size)], 1); lat = X[:, 1]
    n = np.tile([0, 0, 1.0], (len(X), 1)); t = np.tile([0, 1.0, 0], (len(X), 1)) * np.sign(lat + 1e-12)[:, None]
    return dict(X=X, lat=lat, half=np.full(len(X), w), n=n, t_lat=t, r=lat / w, u=X[:, 0], L=float(np.linalg.norm(X.max(0) - X.min(0))))


def test_parse_direction_and_abstain():
    assert parse("Leaves curl upwards and fall off.")["direction"] == +1
    assert parse("leaf margins rolled under")["direction"] == -1
    p = parse("Badly infected leaves curl downward at the edges"); assert p["direction"] == -1 and p["r0"] == 0.5
    assert parse("They usually produce curled leaves at their tips")["status"].startswith("abstain")
    assert parse("Developing leaves become severely distorted")["kind"] is None


def test_curl_target_is_isometric_and_directional():
    F = flat_sheet();
    for sg in (+1, -1):
        Y = F["X"] + curl_target(F, sg, 90.0, 0.0)
        med, p95 = pair_strain(F["X"], Y, F["L"], lo=0.01, hi=0.03)
        assert p95 < 0.02, p95                                   # bending without stretching (smoothing adds a little)
        mar = np.abs(F["r"]) > 0.95; mid = np.abs(F["r"]) < 0.05
        dz = Y[mar, 2].mean() - Y[mid, 2].mean(); assert np.sign(dz) == sg
        # uniform-curvature arc turning 90 deg at the margin: chord angle ~ 45 deg
        dl = np.abs(Y[mar, 1]).mean(); ang = np.degrees(np.arctan2(abs(dz), dl)); assert 35 < ang < 50, ang


def test_bilayer_separates_surfaces():
    F = flat_sheet(40, 20); N = len(F["X"])
    from plyfile import PlyElement
    dt = [(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    v = np.zeros(N, dt); v["x"], v["y"], v["z"] = F["X"].T; v["rot_0"] = 1; v["opacity"] = 8
    for k in range(3): v[f"scale_{k}"] = np.log(0.01)
    F.update(v=v)
    sym = (F["X"][:, 0] > 0).astype(float); red = np.tile([1.0, 0, 0], (N, 1))
    with tempfile.TemporaryDirectory() as d:
        rep, n = bilayer.build(F, f"{d}/b.ply", lower_symptom=sym, lower_symptom_rgb=red, delta_rel=0.01)
        from plyfile import PlyData
        w = PlyData.read(f"{d}/b.ply")["vertex"].data; lay = np.load(f"{d}/b.ply.layers.npy")
        assert len(w) == 2 * N and (lay == 0).sum() == N
        z = w["z"]; assert (z[lay == 0] > 0).all() and (z[lay == 1] < 0).all()      # upper above, lower below
        dc0 = w["f_dc_0"]; assert np.allclose(dc0[:N], dc0[:N][0])                    # upper colour untouched
        assert (dc0[N:][sym > 0.5] > dc0[:N][sym > 0.5]).all()                        # symptom only on the lower layer


def test_gall_volume_and_replacement():
    rng = np.random.default_rng(0); u = rng.normal(size=(20000, 3)); X = u / np.linalg.norm(u, axis=1, keepdims=True)   # unit-sphere organ
    dt = [(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")]
    v = np.zeros(len(X), dt); v["x"], v["y"], v["z"] = X.T; v["rot_0"] = 1; v["opacity"] = 8
    E = dict(v=v, X=X, kernel=np.ones(len(X), bool), gall=np.zeros(len(X), bool), cam=dict(cam_obj=np.array([0, 0, 5.0])))
    c = np.array([[0, 0, 1.0]]); D = np.array([0.6])
    with tempfile.TemporaryDirectory() as d:
        rep = gall.build(E, c, c.copy(), D, f"{d}/g.ply", rng=np.random.default_rng(1))
        g = rep["galls"][0]
        assert 0 < g["volume"] < np.pi / 6 * (1.5 * D[0]) ** 3          # bounded by a sphere 1.5x the nominal diameter
        assert rep["n_hidden"] > 0 and g["kernels_replaced"] > 0          # replaced tissue is hidden
        assert 0 < g["undercut_frac"] < 1                                 # a closed body overhangs; a height field cannot
