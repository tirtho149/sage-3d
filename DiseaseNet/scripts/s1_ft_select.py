"""Stage 1 fine-tuning study, selection step (synthetic LOPO only; no real data).
For each arm (KB2D, KB3D) and mode (linear, lastk, lora, full): the config with the best mean (over folds 0, 1)
best-epoch LOPO macro-F1; epochs = round-up of the mean best epoch (+1) to a multiple of 5 (min 5, max 15).
Writes results/stage1/ft_search/{selection.csv, phase_b_gpu0.sh, phase_b_gpu1.sh} (3 seeds each, DEV evaluation).
Also records over-fitting diagnostics (train loss vs val loss at the last epoch, best vs last val macro-F1).
"""
import glob
import json
import math
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/results/stage1/ft_search"
rows = []
for f in glob.glob(f"{D}/*_c*_fold*.json"):
    j = json.load(open(f)); arm, ci, fold = re.match(r"(.+)_c(\d+)_fold(\d)", os.path.basename(f)[:-5]).groups()
    a = j["args"]; h = j["history"]
    rows.append(dict(arm=arm, ci=int(ci), fold=int(fold), mode=a["mode"], lr_head=a["lr_head"], lr_bb=a["lr_bb"], llrd=a["llrd"],
                     ls=a["ls"], best=j["best_val_macro_f1"], best_epoch=j["best_epoch"], last=j["last_val_macro_f1"],
                     train_loss_last=h[-1]["train_loss"], val_loss_last=h[-1]["val_loss"], val_loss_min=min(x["val_loss"] for x in h),
                     sec=j["train_sec"], trainable=j["trainable_params"]))
R = pd.DataFrame(rows)
G = R.groupby(["arm", "ci", "mode", "lr_head", "lr_bb", "llrd", "ls"]).agg(lopo=("best", "mean"), lopo_last=("last", "mean"),
                                                                         ep=("best_epoch", "mean"), tl=("train_loss_last", "mean"),
                                                                         vl=("val_loss_last", "mean"), folds=("fold", "nunique")).reset_index()
G.to_csv(f"{D}/search_summary.csv", index=False)
sel = G[G.folds == 2].sort_values("lopo", ascending=False).groupby(["arm", "mode"]).head(1).copy()
sel["epochs"] = sel.ep.map(lambda e: int(min(15, max(5, 5 * math.ceil((e + 1) / 5)))))
sel.to_csv(f"{D}/selection.csv", index=False); print(sel.round(4).to_string(index=False))
FLAG = {"linear": "--mode linear", "lastk": "--mode lastk --k 2", "lora": "--mode lora --r 8", "full": "--mode full"}
jobs = []
for _, r in sel.iterrows():
    for s in (0, 1, 2):
        tag = f"{r.arm}_{r['mode']}_sel_s{s}"
        cmd = (f"[ -f results/stage1/ft/{tag}.json ] || ~/Desktop/Research/.venv/bin/python -m src.s1.finetune {r.arm} {FLAG[r['mode']]} "
               f"--lr-head {r.lr_head:g} --lr-bb {r.lr_bb:g} --llrd {r.llrd:g} --ls {r.ls:g} --epochs {r.epochs} --seed {s} "
               f"--tag {tag} --out results/stage1/ft >> logs/stage1/ft_phase_b.log 2>&1")
        jobs.append(cmd)
for g in (0, 1):
    with open(f"{D}/phase_b_gpu{g}.sh", "w") as f:
        f.write(f"cd {ROOT}\n" + "\n".join(f"CUDA_VISIBLE_DEVICES={g} {c}" for c in jobs[g::2]) + "\n")
print(len(jobs), "phase-B jobs")
