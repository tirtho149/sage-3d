"""Operator validity on a synthetic flat sheet (sam3d env): each KB operator must produce the geometry it names,
and the attribute detectors must recover it. These tests fix operator/detector parameters independently of
any specimen result."""
import os, sys, json, tempfile
import numpy as np
from plyfile import PlyData, PlyElement
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geometry_editor.deform import run as deform, fps
from metrics.geom_attributes import attributes

DEF = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "geometry", "operator_defaults.json")))


def plane(path, n=40000, seed=0):
    r = np.random.default_rng(seed); xy = r.uniform(-0.5, 0.5, (n, 2)); z = r.normal(0, 1e-4, n)
    dt = [(k, "f4") for k in ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]]
    v = np.zeros(n, dt); v["x"], v["y"], v["z"] = xy[:, 0], xy[:, 1], z; v["opacity"] = 5; v["scale_0"] = v["scale_1"] = np.log(0.004); v["scale_2"] = np.log(0.0008); v["rot_0"] = 1
    PlyData([PlyElement.describe(v, "vertex")]).write(path)


def test_pucker_makes_distinct_blisters(tmp_path):
    p = str(tmp_path / "plane.ply"); plane(p); o = str(tmp_path / "pk.ply")
    r = deform(p, dict(seed=0, mode="sheet", ops=[dict(DEF["pucker"], op="pucker")], iters=200), o)
    a = attributes(p, o); nb = r["oplog"][0]["n_bumps"]
    assert a["puckered"] > 3 * a["distorted"], a                    # energy at blister scale, not a global warp
    assert a["blister_count"] >= 0.5 * nb, (a["blister_count"], nb)  # bumps survive as distinct blisters


def test_distort_is_low_frequency(tmp_path):
    p = str(tmp_path / "plane.ply"); plane(p); o = str(tmp_path / "ds.ply")
    deform(p, dict(seed=0, mode="sheet", ops=[dict(DEF["distort"], op="distort")], iters=200), o)
    a = attributes(p, o); assert a["distorted"] > 3 * a["puckered"], a


def test_identity_is_zero(tmp_path):
    p = str(tmp_path / "plane.ply"); plane(p); a = attributes(p, p)
    assert a["distorted"] < 1e-6 and a["puckered"] < 1e-6 and a["blister_count"] == 0 and abs(a["thickened"] - 1) < 1e-6, a


def test_thicken_only_scales(tmp_path):
    p = str(tmp_path / "plane.ply"); plane(p); o = str(tmp_path / "th.ply")
    deform(p, dict(seed=0, mode="sheet", ops=[dict(DEF["thicken"], op="thicken")], iters=10), o)
    a = attributes(p, o); assert 1.7 < a["thickened"] < 2.3 and a["distorted"] < 1e-3 and a["puckered"] < 1e-3, a


def test_blister_density_scale_invariant(tmp_path):
    """Same operator on a sheet of half the area: blister COUNT should roughly halve, DENSITY should not."""
    import numpy as np
    from plyfile import PlyData, PlyElement
    p = str(tmp_path / "plane.ply"); plane(p, n=40000); q = str(tmp_path / "half.ply")
    v = PlyData.read(p)["vertex"].data.copy(); v = v[v["x"] < 0.0]                                 # half-area sheet, same point spacing
    PlyData([PlyElement.describe(v, "vertex")]).write(q)
    res = []
    for x in (p, q):
        deform(x, dict(seed=0, mode="sheet", ops=[dict(DEF["pucker"], op="pucker")], iters=200), x + ".pk.ply"); res.append(attributes(x, x + ".pk.ply"))
    a, b = res   # L (bbox diagonal) differs between the sheets, so operator scale differs too: compare densities only
    assert 0.5 < a["blister_density"] / max(b["blister_density"], 1e-9) < 2.0, (a["blister_density"], b["blister_density"], a["blister_count"], b["blister_count"])
