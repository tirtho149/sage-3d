"""Stage 1, step 12: open-set evaluation of T0 probes (frozen DINOv2 + logistic regression), replicate 0.

Known = DEV photos of the 20 head classes, minus the 16 that share a near-duplicate group with an open-set photo.
Unknown = data/real_splits/stage1_openset.csv (19 reserved classes, no group shared with any known-class photo).
Scores (higher = more "known"): MSP = max softmax probability; energy = logsumexp of the logits (decision_function).
Metrics: unknown-detection AUROC, FPR at 95% TPR of known, and AURC (area under the selective risk–coverage curve of
closed-set errors on known photos, MSP-ranked). Closed-set accuracy is never reported for unknown labels.
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python experiments/S1/run_openset.py SET [SET ...]
  -> results/stage1/openset/summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from src.s1 import harness as Hn  # noqa: E402
from src.s1.sets import make  # noqa: E402
from src.training.features import extract  # noqa: E402


def fpr_at_tpr(s_known, s_unk, tpr=0.95):
    thr = np.quantile(s_known, 1 - tpr)
    return float((s_unk >= thr).mean())


def aurc(correct, conf):
    o = np.argsort(-conf); err = 1 - correct[o]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def main():
    C = Hn.classes20(); cix = {c: i for i, c in enumerate(C)}
    D = Hn.dev_set(); D = D[D.class_id.isin(C)]
    R = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet", columns=["group", "partition"])
    D = D[~D.group.isin(set(R[R.partition == "openset"].group))]
    U = pd.read_csv(f"{ROOT}/data/real_splits/stage1_openset.csv")
    Xk, yk = extract(D.path.tolist()), D.class_id.map(cix).values; Xu = extract(U.path.tolist())
    rows = []
    for name in sys.argv[1:]:
        S = make(name); S = S[S.class_id.isin(C)]
        m = Hn.probe_fit(extract(S.path.tolist()), S.class_id.map(cix).values)
        Lk, Lu = m.decision_function(Xk), m.decision_function(Xu)
        Pk, Pu = m.predict_proba(Xk), m.predict_proba(Xu)
        r = dict(set=name, n_known=len(yk), n_unknown=len(Xu))
        for sc, (a, b) in {"msp": (Pk.max(1), Pu.max(1)), "energy": (logsumexp(Lk, 1), logsumexp(Lu, 1))}.items():
            r[f"auroc_{sc}"] = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
            r[f"fpr95_{sc}"] = fpr_at_tpr(a, b)
        corr = (m.classes_[Pk.argmax(1)] == yk).astype(float)
        r["closed_acc_known"] = float(corr.mean()); r["aurc_msp"] = aurc(corr, Pk.max(1))
        rows.append(r); print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    os.makedirs(f"{ROOT}/results/stage1/openset", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{ROOT}/results/stage1/openset/summary.csv", index=False)


if __name__ == "__main__":
    main()
