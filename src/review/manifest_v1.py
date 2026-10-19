"""Write the v1 review manifest from material that already exists on disk.

The manifest is the only input of build_package.py; new demos (results/demos/<demo>/<method>/...) are added
by appending rows to a manifest, not by editing code.

Manifest columns
----------------
item_key        unique, human-readable key (never shown to raters)
specimen        specimen id; the REAL reference photo is data/inputs/<specimen>/image.png unless reference_path set
kb_key          key into data/kb/structured_specs.json (disease text shown to raters)
method          arm / method name (hidden from raters; written only to the sealed key)
method_family   analysis family (hidden); blank -> looked up in configs/review/rubric.json analysis.families
provenance      REAL | EDITED-REAL-2D | SYNTHETIC   (hidden; recorded in the key)
kind            2d | 3d
image_paths     ';'-separated image paths (one for 2d; one per view for 3d)
view_ids        ';'-separated view ids (3d)
view_labels     ';'-separated caption per panel shown to raters (geometry only, no method information)
reference_path  path of the REAL photo (blank -> data/inputs/<specimen>/image.png)
role            test | catch
include         1 | 0  (0 rows are kept in the manifest for the record but not put in the package)
notes           free text (hidden)

usage: python src/review/manifest_v1.py [--out results/review_package_v1/manifest.csv]
"""
import argparse
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SPECIMENS = [f"Peach__Leaf_Curl__{i}" for i in ("0006", "0063", "0087")] + \
            [f"Corn__Smut__{i}" for i in ("0012", "0025", "0052")] + \
            [f"Tomato__Leaf_Mold__{i}" for i in ("0015", "0052", "0092")]

# 18-view rig: views 0-7 azimuth 0..315 deg at +20 deg elevation, 8-15 same at -20 deg, 16 = +80 deg, 17 = -80 deg
# (src/rendering/views.py::rig with n_az=8).
# The requested set was 0, 2, 6, 16, 17. Inspection showed that for leaves none of these shows the opposite
# (abaxial) face of the lamina face-on: 2 and 6 are edge-on, 16/17 look along the leaf axis. Criterion (d)
# (upper vs lower surface) needs that face, so view 6 (mirror of view 2) is replaced by view 4 (azimuth 180 deg).
VIEWS = [0, 4, 2, 16, 17]
VIEW_LABELS = {0: "front, +20\u00b0", 4: "back face, 180\u00b0, +20\u00b0",
               2: "side 90\u00b0, +20\u00b0", 6: "side 270\u00b0, +20\u00b0",
               16: "from above, +80\u00b0", 17: "from below, -80\u00b0"}

ARMS_2D = [("B", "results/pilot/{s}/B/seed0.png"),
           ("C_stage", "results/phaseA/{s}/C_stage/seed0.png"),
           ("I", "results/phaseA/{s}/I/seed0.png")]
ARMS_3D = [("D", "results/phaseA/{s}/D_render/rgb/{v:03d}.png"),
           ("E", "results/phaseA/{s}/E/seed0/render/rgb/{v:03d}.png"),
           ("F", "results/phaseA/{s}/F/seed0/render/rgb/{v:03d}.png"),
           ("G", "results/phaseA/{s}/G/seed0/render/rgb/{v:03d}.png"),
           ("J", "results/phaseA/{s}/J/seed0/render/rgb/{v:03d}.png"),
           ("K", "results/phaseA/{s}/K/seed0/render/rgb/{v:03d}.png"),
           ("H", "results/phaseA/{s}/H/seed0/render/rgb/{v:03d}.png")]
# F (GaussianEditor + KB prompt) is listed but excluded from v1 to keep a rating session under ~60 min;
# set include=1 to add it (adds 9 items, ~6 min).
EXCLUDE_V1 = {"F": "excluded from v1 to keep session length <= 60 min"}
CATCH_ARM = ("C_mismatch", "results/pilot/{s}/C_mismatch/seed0.png")
# Catch items were chosen by visual inspection (2026-10-14): the Peach and Tomato wrong-disease edits visibly remove
# the described symptoms and add others. The Corn wrong-disease edits are almost identical to the input photo, which
# already shows galls, so they cannot work as attention checks and are not used.
CATCH_SPECIMENS = ["Peach__Leaf_Curl__0063", "Peach__Leaf_Curl__0087", "Tomato__Leaf_Mold__0052"]

FIELDS = ["item_key", "specimen", "kb_key", "method", "method_family", "provenance", "kind", "image_paths",
          "view_ids", "view_labels", "reference_path", "role", "include", "notes"]


def kb_key_for(specimen):
    if specimen == "Citrus__Canker__0004":
        return "Citrus__Canker__leaf"
    return specimen.rsplit("__", 1)[0]


def rows():
    out, missing = [], []
    for s in SPECIMENS:
        ref = f"data/inputs/{s}/image.png"
        if not os.path.exists(os.path.join(ROOT, ref)):
            missing.append(ref)
            continue
        for arm, pat in ARMS_2D:
            p = pat.format(s=s)
            if not os.path.exists(os.path.join(ROOT, p)):
                missing.append(p)
                continue
            out.append(dict(item_key=f"{s}|{arm}|seed0", specimen=s, kb_key=kb_key_for(s), method=arm,
                            method_family="", provenance="EDITED-REAL-2D", kind="2d", image_paths=p, view_ids="",
                            view_labels="edited photograph", reference_path=ref, role="test", include=1, notes="seed0"))
        for arm, pat in ARMS_3D:
            ps = [pat.format(s=s, v=v) for v in VIEWS]
            if not all(os.path.exists(os.path.join(ROOT, p)) for p in ps):
                missing.append(pat.format(s=s, v=0))
                continue
            inc = 0 if arm in EXCLUDE_V1 else 1
            out.append(dict(item_key=f"{s}|{arm}|seed0", specimen=s, kb_key=kb_key_for(s), method=arm,
                            method_family="", provenance="SYNTHETIC", kind="3d", image_paths=";".join(ps),
                            view_ids=";".join(str(v) for v in VIEWS),
                            view_labels=";".join(VIEW_LABELS[v] for v in VIEWS), reference_path=ref, role="test",
                            include=inc, notes=EXCLUDE_V1.get(arm, "seed0; rig views 0,4,2,16,17")))
    for s in CATCH_SPECIMENS:
        p = CATCH_ARM[1].format(s=s)
        if not os.path.exists(os.path.join(ROOT, p)):
            missing.append(p)
            continue
        out.append(dict(item_key=f"{s}|C_mismatch|seed0", specimen=s, kb_key=kb_key_for(s), method="C_mismatch",
                        method_family="catch", provenance="EDITED-REAL-2D", kind="2d", image_paths=p, view_ids="",
                        view_labels="edited photograph", reference_path=f"data/inputs/{s}/image.png", role="catch",
                        include=1, notes="attention check: 2D edit prompted with a different disease"))
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/review_package_v1/manifest.csv")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rs, missing = rows()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rs)
    n_inc = sum(int(r["include"]) for r in rs)
    print(f"wrote {out}: {len(rs)} rows, {n_inc} included; {len(missing)} expected items missing on disk:")
    for m in missing:
        print("  missing:", m)


if __name__ == "__main__":
    main()
