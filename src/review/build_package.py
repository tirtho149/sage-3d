"""Build a blinded expert-review package from a manifest (see src/review/manifest_v1.py for the columns).

Output layout (default results/review_package_v1/):
    manifest.csv                 input (not modified by this script)
    key_DO_NOT_SHARE.csv         sealed key: anonymised item id -> method, provenance, sources, duplicate links
    build_meta.json              seed, order, counts, hashes (also sealed; do not share)
    rater_package/               the ONLY folder to send to raters
        index.html               offline rating form (static HTML+JS, exports CSV)
        items/<item_id>.png      one composite per item: REAL reference photo + "candidate" panel(s)
        rating_sheet.csv         one row per item, empty score columns (fallback to the HTML form)
        README_for_raters.md     instructions; no method names

Blinding: raters see an anonymised item id, the specimen's disease description (verbatim KB source quote) and
images labelled only "REAL photo" and "candidate". Method, arm, provenance and duplicate status appear only in
the sealed key. The modality (single edited photograph vs multi-view render) cannot be hidden.

No expert ratings exist yet. This script only prepares the package.

usage: python src/review/build_package.py [--manifest results/review_package_v1/manifest.csv]
                                          [--out results/review_package_v1] [--seed 20261014]
                                          [--dup-frac 0.12] [--min-dup-gap 20] [--sec-per-item 35]
"""
import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
import random
import string

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROVENANCES = {"REAL", "EDITED-REAL-2D", "SYNTHETIC"}
SCORE_COLS = ["a_symptom_match", "b_distribution", "c_geometry", "d_surface", "e_artefacts", "f_overall_accept",
              "prohibited_feature", "comment"]
SHEET_COLS = ["package_id", "rater_id", "item_id", "page"] + SCORE_COLS + ["seconds_on_item", "saved_at"]
KEY_COLS = ["item_id", "page", "item_key", "specimen", "kb_key", "method", "method_family", "provenance", "kind",
            "role", "is_duplicate", "duplicate_of_item_id", "image_paths", "view_ids", "reference_path", "notes"]
DISEASE_NAMES = {"Peach__Leaf_Curl": "Peach leaf curl", "Corn__Smut": "Corn (maize) common smut",
                 "Tomato__Leaf_Mold": "Tomato leaf mold", "Citrus__Canker": "Citrus canker",
                 "Citrus__Canker__leaf": "Citrus canker", "Mango__Anthracnose": "Mango anthracnose"}
CELL = 340          # 3D view cell (px)
CELL_2D = 520       # 2D panel cell (px)
LABEL_H = 34
PAD = 10


# ----------------------------------------------------------------------------------------------- inputs
def p_abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out, errors = [], []
    seen = set()
    for i, r in enumerate(rows):
        if str(r.get("include", "1")).strip() in ("0", "false", "False", "no"):
            continue
        r = {k: (v or "").strip() for k, v in r.items()}
        if r["item_key"] in seen:
            errors.append(f"row {i}: duplicate item_key {r['item_key']}")
        seen.add(r["item_key"])
        if r["provenance"] not in PROVENANCES:
            errors.append(f"row {i}: provenance {r['provenance']!r} not in {sorted(PROVENANCES)}")
        if r["kind"] not in ("2d", "3d"):
            errors.append(f"row {i}: kind must be 2d or 3d")
        r["role"] = r.get("role") or "test"
        if r["role"] not in ("test", "catch"):
            errors.append(f"row {i}: role must be test or catch")
        r["reference_path"] = r.get("reference_path") or f"data/inputs/{r['specimen']}/image.png"
        paths = [p for p in r["image_paths"].split(";") if p]
        if not paths:
            errors.append(f"row {i}: no image_paths")
        for p in paths + [r["reference_path"]]:
            if not os.path.exists(p_abs(p)):
                errors.append(f"row {i}: missing file {p}")
        out.append(r)
    if errors:
        raise SystemExit("manifest errors:\n  " + "\n  ".join(errors))
    return out


def family_of(r, families):
    if r.get("method_family"):
        return r["method_family"]
    for fam, methods in families.items():
        if r["method"] in methods:
            return fam
    return "other"


def disease_text(kb, key):
    spec = kb[key]
    s = spec["stage"]
    cur = list(dict.fromkeys(p["sentence"] for p in spec["phrases"] if p["stage"] == s))
    other = list(dict.fromkeys(p["sentence"] for p in spec["phrases"] if p["stage"] != s))
    return dict(disease=DISEASE_NAMES.get(key, key.replace("__", " ").replace("_", " ")),
                organ=spec.get("target_organ", ""), quote=spec["source_quote"], source_url=spec.get("source_url", ""),
                stage_index=s + 1, n_stages=spec.get("n_stages", 1), stage_sentences=cur,
                other_stage_sentences=other)


# ----------------------------------------------------------------------------------------------- order
def make_order(items, rng, dup_frac, min_gap):
    """Shuffle (no two consecutive items from the same specimen where possible), then re-present a random
    subset of test items at least min_gap positions after their first presentation."""
    best = None
    for _ in range(200):
        seq = items[:]
        rng.shuffle(seq)
        clashes = sum(seq[i]["specimen"] == seq[i + 1]["specimen"] for i in range(len(seq) - 1))
        if best is None or clashes < best[0]:
            best = (clashes, seq)
        if clashes == 0:
            break
    seq = [dict(r, is_duplicate=0, duplicate_of="") for r in best[1]]
    n_test = sum(r["role"] == "test" for r in seq)
    n_dup = int(round(dup_frac * n_test))
    eligible = [i for i, r in enumerate(seq) if r["role"] == "test" and i <= len(seq) - min_gap]
    # stratify duplicates over 2d/3d so both modalities have test-retest pairs
    by_kind = {}
    for i in eligible:
        by_kind.setdefault(seq[i]["kind"], []).append(i)
    picks = []
    kinds = sorted(by_kind)
    for k in kinds:
        rng.shuffle(by_kind[k])
    while len(picks) < n_dup and any(by_kind.values()):
        for k in kinds:
            if by_kind[k] and len(picks) < n_dup:
                picks.append(by_kind[k].pop())
    originals = [seq[i] for i in sorted(picks)]
    for obj in originals:
        cur = next(j for j, r in enumerate(seq) if r is obj)
        pos = rng.randint(cur + min_gap, len(seq))
        seq.insert(pos, dict(obj, is_duplicate=1, duplicate_of=obj["item_key"]))
    return seq


def make_ids(n, rng):
    alphabet = "ACDEFHJKLMNPRTUVWXY3479"
    ids = set()
    while len(ids) < n:
        ids.add("X-" + "".join(rng.choice(alphabet) for _ in range(4)))
    ids = sorted(ids)
    rng.shuffle(ids)
    return ids


# ----------------------------------------------------------------------------------------------- images
def font(size):
    for p in [os.path.expanduser("~/micromamba/envs/sam3d/lib/python3.11/site-packages/matplotlib/mpl-data/"
                                 "fonts/ttf/DejaVuSans.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        import matplotlib
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def fit(im, w, h, bg=(255, 255, 255)):
    im = im.convert("RGB")
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    c = Image.new("RGB", (w, h), bg)
    c.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return c


def common_scale_crops(ims, thr=245, margin=0.08):
    """Crop each render to a square of the same side around its foreground, so all views share one scale."""
    boxes = []
    for im in ims:
        a = im.convert("L").point(lambda v: 255 if v < thr else 0)
        boxes.append(a.getbbox() or (0, 0, im.width, im.height))
    side = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
    side = int(side * (1 + 2 * margin))
    out = []
    for im, b in zip(ims, boxes):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        box = (int(cx - side / 2), int(cy - side / 2), int(cx - side / 2) + side, int(cy - side / 2) + side)
        padded = Image.new("RGB", (im.width + 2 * side, im.height + 2 * side), (255, 255, 255))
        padded.paste(im.convert("RGB"), (side, side))
        out.append(padded.crop(tuple(v + side for v in box)))
    return out


def labelled(cell, text, colour, fnt):
    w, h = cell.size
    c = Image.new("RGB", (w + 6, h + LABEL_H + 6), colour)
    d = ImageDraw.Draw(c)
    d.text((8, 6), text, fill=(255, 255, 255), font=fnt)
    c.paste(cell, (3, LABEL_H + 3))
    return c


def composite(r, item_id):
    f_lab, f_head = font(17), font(20)
    REALC, CANDC = (20, 110, 50), (90, 90, 90)
    ref = Image.open(p_abs(r["reference_path"]))
    paths = [p for p in r["image_paths"].split(";") if p]
    labels = [s for s in r.get("view_labels", "").split(";") if s] or [""] * len(paths)
    if r["kind"] == "2d":
        cells = [labelled(fit(ref, CELL_2D, CELL_2D), "REAL photo (reference, unedited)", REALC, f_lab),
                 labelled(fit(Image.open(p_abs(paths[0])), CELL_2D, CELL_2D), "candidate", CANDC, f_lab)]
        cols = 2
    else:
        views = common_scale_crops([Image.open(p_abs(p)) for p in paths])
        cells = [labelled(fit(ref, CELL, CELL), "REAL photo (reference)", REALC, f_lab)]
        for v, lab in zip(views, labels):
            cells.append(labelled(fit(v, CELL, CELL), f"candidate: {lab}" if lab else "candidate", CANDC, f_lab))
        cols = 3
    cw, ch = cells[0].size
    rows_n = (len(cells) + cols - 1) // cols
    W = cols * cw + (cols + 1) * PAD
    H = 44 + rows_n * ch + (rows_n + 1) * PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 10), f"Item {item_id}", fill=(0, 0, 0), font=f_head)
    for k, c in enumerate(cells):
        x = PAD + (k % cols) * (cw + PAD)
        y = 44 + PAD + (k // cols) * (ch + PAD)
        canvas.paste(c, (x, y))
    return canvas


# ----------------------------------------------------------------------------------------------- outputs
def write_html(path, package_id, rubric, items_js):
    data = json.dumps(dict(package_id=package_id, rubric=rubric, items=items_js), ensure_ascii=False)
    data = data.replace("</", "<\\/")
    tpl = open(os.path.join(os.path.dirname(__file__), "review_form_template.html"), encoding="utf-8").read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/null", data).replace("__PACKAGE_ID__", html.escape(package_id)))


def write_readme(path, package_id, n_items, minutes, diseases, rubric):
    L = [f"# Plant-disease image review: instructions for raters", "",
         f"Package: `{package_id}`  ", f"Number of items: {n_items}  ",
         f"Estimated time: about {minutes} minutes (about 35 seconds per item after the first few, "
         f"plus about 8 minutes to read these instructions). You may pause and resume.", "",
         "## What you are asked to do", "",
         "Each item shows a REAL photograph of a plant organ (green label) and a *candidate* image of the same "
         "specimen (grey label). A candidate is either a single image or several views of the same organ from "
         "different directions (front, back face, side, from above, from below). Candidates were produced in different "
         "ways; some may be unaltered. You are not told how any candidate was produced, and you should not try "
         "to guess.", "",
         "The REAL photo may already show symptoms of the disease; this is expected. Judge whether the candidate is a "
         "realistic example of the described disease at the stated stage, whatever the REAL photo shows.", "",
         "For each candidate, judge it against the written disease description shown on the item page "
         "(quoted verbatim from an extension or teaching source) for the stated stage. Judge the candidate, "
         "not the REAL photo; the REAL photo is there so you can see the specimen before any change.", "",
         "Some items appear more than once. Rate each appearance independently; do not try to remember earlier "
         "scores.", "",
         "## How to rate", "",
         "1. Open `index.html` in a web browser (Chrome, Firefox, Edge or Safari). No internet connection is "
         "needed; nothing is sent anywhere.",
         "2. Enter your rater code (given to you by the study coordinator; do not use your name).",
         "3. For each item, choose a score from 1 to 5 for each of the six criteria (or NA if the criterion "
         "cannot be judged from the images), answer the yes/no question, and add a comment if you wish.",
         "4. Click the image to open it at full size.",
         "5. Your answers are saved in this browser as you go. When finished, click **Export CSV** and send the "
         "downloaded file to the study coordinator. You can also export part-way through as a backup.",
         "6. If the HTML form does not work, fill in `rating_sheet.csv` directly (one row per item; item images "
         "are in `items/`).", "",
         "## Criteria", ""]
    for s in rubric["scales"]:
        L += [f"### {s['short']}", "", s["question"], ""]
        L += [f"- **{k}**: {v}" for k, v in s["anchors"].items()]
        L.append("")
    for b in rubric["binary"]:
        L += [f"### {b['short']} (yes / no)", "", b["question"], ""]
    L += ["NA: " + rubric["na_definition"], "",
          "## Disease descriptions used in this package", ""]
    for d in diseases:
        L += [f"### {d['disease']} ({d['organ']})", "", f"> {d['quote']}", "",
              f"Source: {d['source_url']}", "",
              f"Stage to judge (stage {d['stage_index']} of {d['n_stages']}): " + " ".join(d["stage_sentences"]), ""]
        if d["other_stage_sentences"]:
            L += ["Should not be present at this stage (later or other stage): " +
                  " ".join(d["other_stage_sentences"]), ""]
    L += ["## Questions", "", "Contact the study coordinator. Please do not discuss items with other raters "
          "until all ratings have been returned.", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/review_package_v1/manifest.csv")
    ap.add_argument("--out", default="results/review_package_v1")
    ap.add_argument("--rubric", default="configs/review/rubric.json")
    ap.add_argument("--kb", default="data/kb/structured_specs.json")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--dup-frac", type=float, default=0.12)
    ap.add_argument("--min-dup-gap", type=int, default=20)
    ap.add_argument("--sec-per-item", type=float, default=35.0)
    ap.add_argument("--intro-min", type=float, default=8.0)
    ap.add_argument("--package-id", default=None)
    a = ap.parse_args()

    rubric = json.load(open(p_abs(a.rubric)))
    kb = json.load(open(p_abs(a.kb)))
    rows = read_manifest(p_abs(a.manifest))
    for r in rows:
        if r["kb_key"] not in kb:
            raise SystemExit(f"kb_key {r['kb_key']} not in {a.kb}")
        r["method_family"] = family_of(r, rubric["analysis"]["families"])
    rng = random.Random(a.seed)
    seq = make_order(rows, rng, a.dup_frac, a.min_dup_gap)
    ids = make_ids(len(seq), rng)
    first_id = {}
    for i, r in enumerate(seq):
        r["item_id"], r["page"] = ids[i], i + 1
        if not r["is_duplicate"]:
            first_id[r["item_key"]] = r["item_id"]
    for r in seq:
        r["duplicate_of_item_id"] = first_id[r["item_key"]] if r["is_duplicate"] else ""

    mhash = sha(p_abs(a.manifest))[:10]
    package_id = a.package_id or f"{os.path.basename(os.path.normpath(a.out))}_{mhash}"
    out = p_abs(a.out)
    rp = os.path.join(out, "rater_package")
    os.makedirs(os.path.join(rp, "items"), exist_ok=True)

    items_js = []
    for r in seq:
        im = composite(r, r["item_id"])
        im.save(os.path.join(rp, "items", f"{r['item_id']}.png"), optimize=True)
        d = disease_text(kb, r["kb_key"])
        items_js.append(dict(id=r["item_id"], page=r["page"], img=f"items/{r['item_id']}.png", **d))

    with open(os.path.join(out, "key_DO_NOT_SHARE.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in seq:
            w.writerow(dict(r, duplicate_of_item_id=r["duplicate_of_item_id"]))
    with open(os.path.join(rp, "rating_sheet.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS)
        w.writeheader()
        for r in seq:
            w.writerow(dict(package_id=package_id, item_id=r["item_id"], page=r["page"]))
    n = len(seq)
    minutes = int(round(n * a.sec_per_item / 60 + a.intro_min))
    rub_public = {k: rubric[k] for k in ("rubric_id", "version", "scales", "binary", "free_text", "na_code",
                                         "na_definition")}
    write_html(os.path.join(rp, "index.html"), package_id, rub_public, items_js)
    diseases = []
    for key in dict.fromkeys(r["kb_key"] for r in seq):
        diseases.append(disease_text(kb, key))
    write_readme(os.path.join(rp, "README_for_raters.md"), package_id, n, minutes, diseases, rubric)

    counts = {}
    for r in seq:
        lab = "duplicate" if r["is_duplicate"] else r["role"]
        counts[lab] = counts.get(lab, 0) + 1
    by_method = {}
    for r in seq:
        if r["role"] == "test" and not r["is_duplicate"]:
            by_method[r["method"]] = by_method.get(r["method"], 0) + 1
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(rp) for fn in fns)
    meta = dict(package_id=package_id, built_at=dt.datetime.now().isoformat(timespec="seconds"), seed=a.seed,
                manifest=a.manifest, manifest_sha256=sha(p_abs(a.manifest)), rubric=a.rubric,
                rubric_sha256=sha(p_abs(a.rubric)), dup_frac=a.dup_frac, min_dup_gap=a.min_dup_gap,
                n_items_total=n, counts=counts, unique_test_items_by_method=by_method,
                est_minutes=minutes, sec_per_item=a.sec_per_item, intro_min=a.intro_min,
                rater_package_bytes=size, order=[r["item_id"] for r in seq],
                ratings_status="NO REAL RATINGS EXIST YET (package built, not yet distributed)")
    json.dump(meta, open(os.path.join(out, "build_meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("package_id", "n_items_total", "counts", "unique_test_items_by_method",
                                           "est_minutes", "rater_package_bytes")}, indent=1))


if __name__ == "__main__":
    main()
