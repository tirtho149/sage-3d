"""Stage 1, step 2: independent-asset diversity of the Stage-0 synthetic data (CPU; cached DINOv2 ViT-S/14 features).

Per class x arm: images, independent root photos, assets, views, conditions, backgrounds; embedding spread
(mean pairwise cosine distance), share of embedding variance between parents (parent_var_share), near-duplicate rate (pairs with cosine >= NEAR), number of embedding clusters
(average-linkage at cosine distance CLUST) and the effective number of independent parents (exp of the entropy of
the between-parent variance share; = #parents when parents are equally distinct, -> 1 when one parent dominates).
Reference: the same statistics on an equal-size random sample of real test photos of the class.
Outputs: results/stage1/diversity/{per_class.csv, summary.json}, figures/stage1/diversity_images_vs_parents.pdf
usage: ~/Desktop/Research/.venv/bin/python scripts/s1_diversity.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.training.features import load_cache  # noqa: E402

NEAR, CLUST = 0.95, 0.10
OUT = f"{ROOT}/results/stage1/diversity"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{ROOT}/figures/stage1", exist_ok=True)


def stats(F, parents):
    F = F.astype(np.float32); F /= np.linalg.norm(F, axis=1, keepdims=True)
    S = F @ F.T; iu = np.triu_indices(len(F), 1); s = S[iu]
    Z = linkage(F, "average", metric="cosine"); ncl = len(set(fcluster(Z, CLUST, "distance")))
    mu = F.mean(0); P = pd.Series(parents)
    w = np.array([((F[P.values == p].mean(0) - mu) ** 2).sum() * (P.values == p).sum() for p in P.unique()])
    q = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    tot = ((F - mu) ** 2).sum()
    return dict(spread=float(1 - s.mean()), near_dup_rate=float((s >= NEAR).mean()), clusters=ncl,
                eff_parents=float(np.exp(-(q * np.log(q + 1e-12)).sum())), parent_var_share=float(w.sum() / tot))


def main():
    cache = load_cache()
    syn = pd.concat([pd.read_csv(f"{ROOT}/data/synthetic/stage0/{m}") for m in ("manifest_3d.csv", "manifest_2d.csv")])
    real = pd.read_csv(f"{ROOT}/data/real_splits/stage0_test.csv")
    E = pd.read_csv(f"{ROOT}/CLASS_ELIGIBILITY.csv").set_index("class_id")
    rows = []
    for (c, a), g in syn.groupby(["class_id", "arm"]):
        F = np.stack([cache[p] for p in g.path])
        r = dict(class_id=c, arm=a, images=len(g), root_photos=g.root_sha1.nunique(), assets=g.asset_id.nunique(),
                 views=g.view.nunique(), conditions=g.cond.nunique(), backgrounds=int(g.bg_asset.fillna(g.asset_id).nunique()),
                 stages=1, real_parent_pool_groups=int(E.loc[c, "groups_parent"]))
        r.update(stats(F, g.root_sha1.values)); rows.append(r)
    rng = np.random.default_rng(0)
    for c, g in real.groupby("class_id"):
        g = g.sample(min(96, len(g)), random_state=0)
        r = dict(class_id=c, arm="REAL(test sample)", images=len(g), root_photos=len(g), assets=0, views=0, conditions=0, backgrounds=0, stages=0,
                 real_parent_pool_groups=int(E.loc[c, "groups_parent"]))
        r.update(stats(np.stack([cache[p] for p in g.path]), g.group.values)); rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(f"{OUT}/per_class.csv", index=False)
    S = D.groupby("arm")[["images", "root_photos", "spread", "near_dup_rate", "clusters", "eff_parents", "parent_var_share"]].mean().round(3)
    print(S.to_string())
    lowdiv = D[(D.arm != "REAL(test sample)") & (D.near_dup_rate > 0.5)][["class_id", "arm", "near_dup_rate", "clusters"]]
    json.dump(dict(near_threshold=NEAR, cluster_distance=CLUST, by_arm=S.reset_index().to_dict("records"),
                   classes_mostly_near_identical=lowdiv.to_dict("records")), open(f"{OUT}/summary.json", "w"), indent=1)
    print("class x arm with >50% near-identical pairs:", len(lowdiv)); print(lowdiv.to_string())
    # figure: per class, nominal images vs independent parents (used / effective / available); variance share by arm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    U = D[D.arm == "KB3D"].sort_values("real_parent_pool_groups").reset_index(drop=True); yy = np.arange(len(U))
    ax[0].hlines(yy, U.root_photos, U.real_parent_pool_groups, color="#dddddd", lw=1.2, zorder=1)
    ax[0].scatter(U.images, yy, marker="s", s=14, color="#707372", label="synthetic images / class (one arm)", zorder=3)
    ax[0].scatter(U.root_photos, yy, s=16, color="#C8102E", label="independent parents used", zorder=4)
    ax[0].scatter(U.eff_parents, yy, s=10, facecolors="none", edgecolors="#C8102E", label="effective parents (embedding)", zorder=4)
    ax[0].scatter(U.real_parent_pool_groups, yy, s=12, color="black", marker="|", label="independent parents available", zorder=3)
    ax[0].set_yticks(yy, [c.replace("::", ": ").replace("_", " ") for c in U.class_id], fontsize=5.4); ax[0].set_xscale("log")
    ax[0].set_xlabel("count per class (log)"); ax[0].legend(frameon=False, fontsize=5.6, loc="lower right")
    ax[0].set_title("Stage 0: 3 parents per class, 96 images per class and arm", fontsize=7)
    B = D.groupby("arm").parent_var_share.mean().reindex(["REAL(test sample)", "PHOTO", "KB2D", "U3D", "KB3D"])
    ax[1].barh(np.arange(len(B)), B.values, color=["black"] + ["#707372"] * 4, height=0.6)
    ax[1].set_yticks(np.arange(len(B)), ["real photos\n(test sample)", "PHOTO", "KB2D*", "U3D", "KB3D*"], fontsize=6)
    for k, v in enumerate(B.values):
        ax[1].text(v + 0.01, k, f"{v:.2f}", va="center", fontsize=6)
    ax[1].set_xlim(0, 1.15); ax[1].set_xlabel("share of embedding variance\nbetween independent parents")
    ax[1].set_title("Where synthetic diversity comes from", fontsize=7)
    fig.text(0.99, 0.01, "* Stage-0 arms as generated (edit bug: KB2D = photo, KB3D = photo-textured 3D)", ha="right", fontsize=5.2, color="#707372")
    for a in ax:
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{ROOT}/figures/stage1/diversity_images_vs_parents.pdf"); fig.savefig(f"{ROOT}/figures/stage1/diversity_images_vs_parents.png", dpi=200)

if __name__ == "__main__":
    main()
