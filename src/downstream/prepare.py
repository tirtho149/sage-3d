"""Stage 1 of the minimal downstream test: select real images, extract them from the SAGE shards,
build the synthetic image sets (with the pre-registered compositing), and run leakage checks.

Run with an interpreter that has pyarrow + imagehash (e.g. ~/Desktop/Research/.venv/bin/python):
    python -m src.downstream.prepare --work <scratch_dir>
Writes results/downstream_minimal/image_manifest.csv and leakage.json. Image files go to <work>
(temporary; deleted after feature extraction).
"""
import argparse, hashlib, io, json
from pathlib import Path

import numpy as np, pandas as pd, pyarrow.parquet as pq, imagehash
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

from .common import (ROOT, OUT, SHARDS, CROPS, PILOTS, N_EVAL_CAP, N_TRAIN_CAP, N_BG, VIEWS,
                     SYN_SOURCES, sid_to_specimen)

CANVAS, OBJ_FRAC, BLUR = 256, 0.70, 8


def phash_int(img):
    return int(str(imagehash.phash(img.convert("RGB"))), 16)


def ham(a, b):
    """a: (n,) uint64, b: (m,) uint64 -> (n,m) Hamming distances."""
    x = a[:, None] ^ b[None, :]
    return np.unpackbits(x.view(np.uint8).reshape(len(a), len(b), 8), axis=2).sum(2)


def select_real():
    spec = pd.read_csv(ROOT / "data/metadata/specimen_manifest.csv")
    rows = []
    for crop, (pilot, others) in CROPS.items():
        s = pd.read_csv(ROOT / f"data/metadata/splits_{crop}.csv")
        s = s[s.disease.isin([pilot] + others)].copy()
        sp = spec[spec.specimen_id.isin([sid_to_specimen(x) for x in PILOTS[crop]])]
        assert len(sp) == 3, (crop, len(sp))
        pilot_clusters = set(s[s.sha1.isin(sp.sha1)].cluster)
        assert len(pilot_clusters) == 3, pilot_clusters
        rng = np.random.default_rng(0)
        for dis, g in s.groupby("disease"):
            ev = g[g.split.isin(["test", "val"])]
            ev = ev.sample(min(len(ev), N_EVAL_CAP), random_state=int(rng.integers(1 << 31)))
            tr = g[(g.split == "build") & ~g.cluster.isin(pilot_clusters)]
            tr = tr.sample(min(len(tr), N_TRAIN_CAP), random_state=int(rng.integers(1 << 31)))
            bg = g[g.split == "opt"]
            bg = bg.sample(min(len(bg), N_BG), random_state=int(rng.integers(1 << 31)))
            for role, d in (("eval", ev), ("train", tr), ("bg", bg)):
                d = d.assign(role=role, is_pilot_class=(dis == pilot))
                rows.append(d)
        # pilot source photos themselves (for condition S and leakage checks)
        src = s[s.sha1.isin(sp.sha1)].assign(role="pilot_source", is_pilot_class=True)
        rows.append(src)
    return pd.concat(rows, ignore_index=True)


def extract(sel, work):
    work.mkdir(parents=True, exist_ok=True)
    paths, phs = {}, {}
    for shard, g in sel.groupby("shard"):
        t = pq.read_table(SHARDS / shard, columns=["crop", "filename", "image"]).to_pandas()
        t = t[t.crop.isin(g.crop.unique())]
        want = set(g.sha1)
        for crop, fn, im in zip(t.crop, t.filename, t.image):
            b = im["bytes"]
            h = hashlib.sha1(b).hexdigest()
            if h not in want or h in paths:
                continue
            img = Image.open(io.BytesIO(b)).convert("RGB")
            img = ImageOps.contain(img, (512, 512)) if max(img.size) > 512 else img
            p = work / "real" / f"{h}.jpg"
            p.parent.mkdir(exist_ok=True)
            img.save(p, quality=95)
            paths[h] = str(p); phs[h] = phash_int(img)
        print(shard, len(g), "found so far", len(paths), flush=True)
    sel = sel.copy()
    sel["path"] = sel.sha1.map(paths)
    sel["phash"] = sel.sha1.map(phs)
    missing = sel.path.isna().sum()
    print("missing", missing)
    return sel[sel.path.notna()]


def composite(render_rgb, alpha, bg_img, white=False):
    rgb = Image.open(render_rgb).convert("RGB")
    a = Image.open(alpha).convert("L")
    bbox = a.point(lambda v: 255 if v > 8 else 0).getbbox() or (0, 0, *a.size)
    rgb, a = rgb.crop(bbox), a.crop(bbox)
    s = OBJ_FRAC * CANVAS / max(rgb.size)
    sz = (max(1, round(rgb.size[0] * s)), max(1, round(rgb.size[1] * s)))
    rgb, a = rgb.resize(sz, Image.BICUBIC), a.resize(sz, Image.BICUBIC)
    if white:
        canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    else:
        canvas = ImageOps.fit(bg_img, (CANVAS, CANVAS)).filter(ImageFilter.GaussianBlur(BLUR))
    canvas.paste(rgb, ((CANVAS - sz[0]) // 2, (CANVAS - sz[1]) // 2), a)
    return canvas


AUGS = ["id", "hflip", "rot+15", "rot-15", "crop0.7", "jitter"]


def classical_aug(img, name):
    if name == "id": return img
    if name == "hflip": return ImageOps.mirror(img)
    if name == "rot+15": return img.rotate(15, resample=Image.BICUBIC, expand=False)
    if name == "rot-15": return img.rotate(-15, resample=Image.BICUBIC, expand=False)
    if name == "crop0.7":
        w, h = img.size; cw, ch = int(w * .7), int(h * .7)
        return img.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    if name == "jitter":
        img = ImageEnhance.Brightness(img).enhance(1.15)
        img = ImageEnhance.Contrast(img).enhance(0.85)
        return ImageEnhance.Color(img).enhance(1.2)
    raise ValueError(name)


def build_synthetic(real, work):
    rows = []
    for crop, sids in PILOTS.items():
        bgs = real[(real.crop == crop) & (real.role == "bg")]
        rng = np.random.default_rng(1)
        # pre-registered: background class uniform over the crop's selected classes, fixed per (sid, slot)
        classes = sorted(bgs.disease.unique())
        for si, sid in enumerate(sids):
            slot_bg = []
            for slot in range(len(VIEWS)):
                dis = classes[int(rng.integers(len(classes)))]
                cand = bgs[bgs.disease == dis]
                slot_bg.append(cand.iloc[int(rng.integers(len(cand)))].path)
            # S: source photo + classical augs
            inp = ROOT / "data/inputs" / sid / "image.png"
            src = Image.open(inp).convert("RGB")
            for a in AUGS:
                p = work / "syn" / "S" / f"{sid}_{a}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                im = classical_aug(src, a); im.save(p, quality=95)
                rows.append(dict(crop=crop, sid=sid, cond="S", bgmode="natural", slot=a, path=str(p), phash=phash_int(im)))
            # KB2D: C_stage seeds 0-2 + C seeds 0-2 (already natural background)
            for kind, base in (("C_stage", ROOT / "results/phaseA" / sid / "C_stage"), ("C", ROOT / "results/pilot" / sid / "C")):
                for sd in range(3):
                    f = base / f"seed{sd}.png"
                    im = Image.open(f).convert("RGB")
                    rows.append(dict(crop=crop, sid=sid, cond="KB2D", bgmode="natural", slot=f"{kind}_seed{sd}",
                                     path=str(f), phash=phash_int(im)))
            # 3D render conditions
            for cond, cands in SYN_SOURCES.items():
                src_dir = None
                for c in cands:
                    d = ROOT / "results/phaseA" / sid / (c if c == "D_render" else f"{c}/seed0/render")
                    if (d / "rgb" / "000.png").exists():
                        src_dir, used = d, c; break
                if src_dir is None:
                    continue
                for slot, v in enumerate(VIEWS):
                    rgbp, ap = src_dir / "rgb" / f"{v:03d}.png", src_dir / "alpha" / f"{v:03d}.png"
                    bg = Image.open(slot_bg[slot]).convert("RGB")
                    for mode in ("natural", "white"):
                        im = composite(rgbp, ap, bg, white=(mode == "white"))
                        p = work / "syn" / cond / mode / f"{sid}_{v:03d}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
                        im.save(p, quality=95)
                        rows.append(dict(crop=crop, sid=sid, cond=cond, bgmode=mode, slot=f"view{v:03d}", source_arm=used,
                                         path=str(p), phash=phash_int(im), render=str(rgbp.relative_to(ROOT))))
    return pd.DataFrame(rows)


def leakage(real, syn):
    out = {}
    for crop in CROPS:
        r = real[real.crop == crop]
        ev = np.array(r[r.role == "eval"].phash.astype("uint64").values, dtype=np.uint64)
        tr = np.array(r[r.role == "train"].phash.astype("uint64").values, dtype=np.uint64)
        src = r[r.role == "pilot_source"]
        inp = np.array([phash_int(Image.open(ROOT / "data/inputs" / s / "image.png")) for s in PILOTS[crop]], dtype=np.uint64)
        kb2d = np.array(syn[(syn.crop == crop) & (syn.cond == "KB2D")].phash.astype("uint64").values, dtype=np.uint64)
        d_in_ev, d_tr_ev, d_kb_ev = ham(inp, ev), ham(tr, ev), ham(kb2d, ev)
        d_in_tr = ham(inp, tr)
        out[crop] = dict(
            n_eval=int(len(ev)), n_train_pool=int(len(tr)),
            pilot_source_sha1_in_eval=int(src.sha1.isin(r[r.role == "eval"].sha1).sum()),
            pilot_source_sha1_in_train_pool=int(src.sha1.isin(r[r.role == "train"].sha1).sum()),
            pilot_input_vs_eval_ham_le6=int((d_in_ev <= 6).sum()), pilot_input_vs_eval_ham_le10=int((d_in_ev <= 10).sum()),
            pilot_input_vs_eval_min_ham=int(d_in_ev.min()),
            pilot_input_vs_trainpool_ham_le6=int((d_in_tr <= 6).sum()), pilot_input_vs_trainpool_ham_le10=int((d_in_tr <= 10).sum()),
            trainpool_vs_eval_pairs_ham_le6=int((d_tr_ev <= 6).sum()), trainpool_vs_eval_pairs_ham_le10=int((d_tr_ev <= 10).sum()),
            trainpool_vs_eval_min_ham=int(d_tr_ev.min()),
            kb2d_vs_eval_ham_le6=int((d_kb_ev <= 6).sum()), kb2d_vs_eval_ham_le10=int((d_kb_ev <= 10).sum()),
        )
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); a = ap.parse_args()
    work = Path(a.work)
    sel = select_real()
    spec = pd.read_csv(ROOT / "data/metadata/specimen_manifest.csv")[["sha1", "specimen_id"]]
    sel = sel.merge(spec, on="sha1", how="left")
    print(sel.groupby(["crop", "disease", "role"]).size().unstack(fill_value=0))
    real = extract(sel, work)
    syn = build_synthetic(real, work)
    leak = leakage(real, syn)
    print(json.dumps(leak, indent=1))
    real.assign(kind="real").to_csv(OUT / "image_manifest_real.csv", index=False)
    syn.assign(kind="syn").to_csv(OUT / "image_manifest_syn.csv", index=False)
    json.dump(leak, open(OUT / "leakage.json", "w"), indent=1)
    counts = real.groupby(["crop", "disease", "role"]).size().unstack(fill_value=0)
    counts.to_csv(OUT / "data_counts.csv")
    print(syn.groupby(["crop", "cond", "bgmode"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
