"""Review item B2: apply the observed-evidence policy to every demo and pilot edit.

Reads input_view_iou already stored in results/demos/*/*/metrics.json (no rendering).
TAU = largest fitted-IoU drop of the appearance-only arms E/F (src/geometry_editor/evidence_policy.py).
Each shape edit is classified in preserve mode at TAU and 2*TAU; KB-unspecified edits
(Peach curl hypotheses) are abstentions by protocol and listed separately.

usage: python3 experiments/review/b2_evidence_policy.py
  -> results/review/b2_policy/summary.{json,md}
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
from geometry_editor.evidence_policy import decide, tau_from_appearance_arms  # noqa: E402

OUT = f"{ROOT}/results/review/b2_policy"
SHAPE = {  # demo -> (arm prefix -> human label); appearance-only arms are excluded from judgement
    "mango_curl": {"CU_kb": "KB curl (cited dir.)", "CD_opposite": "reversed curl (control)", "G": "generic geometry"},
    "peach": {"J2_": "KB pucker, evidence-placed", "J2u": "KB pucker, uniform", "J": "KB geometry v1", "G": "generic geometry",
              "J2+curl_up_HYP": "curl up (hypothesis)", "J2+curl_down_HYP": "curl down (hypothesis)"},
    "corn": {"V_": "volumetric galls on kernels", "Vr": "volumetric galls, random sites", "J": "KB height field", "G": "generic geometry"},
    "tomato": {"BLk": "two-surface mold", "BLr": "two-surface, random", "BLu": "two-surface, uniform"},
}
HYP = ("J2+curl_up_HYP", "J2+curl_down_HYP")


def label(demo, arm):
    for pre, lab in sorted(SHAPE.get(demo, {}).items(), key=lambda kv: -len(kv[0])):
        if arm == pre or arm.startswith(pre):
            return lab
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    recs, app = [], []
    for f in sorted(glob.glob(f"{ROOT}/results/demos/*/*/metrics.json")):
        d = json.load(open(f)); demo = f.split("/")[-3]; M = d.get("methods", {})
        base = M.get("D", {}).get("input_view_iou")
        if base is None:
            continue
        for arm, m in M.items():
            if not isinstance(m, dict) or m.get("input_view_iou") is None or arm == "D":
                continue
            if arm in ("E", "F", "K"):
                app.append(base - m["input_view_iou"]); continue
            lab = label(demo, arm)
            if lab:
                recs.append(dict(demo=demo, specimen=d.get("specimen"), arm=arm, label=lab,
                                 iou_before=base, iou_after=m["input_view_iou"]))
    tau = tau_from_appearance_arms(app)
    for r in recs:
        hyp = r["arm"] in HYP
        r["preserve_tau"] = decide(r["iou_before"], r["iou_after"], tau=tau, kb_specifies=not hyp).as_dict()
        r["preserve_2tau"] = decide(r["iou_before"], r["iou_after"], tau=2 * tau, kb_specifies=not hyp).as_dict()
    summ = {}
    for r in recs:
        k = (r["demo"], r["label"]); s = summ.setdefault(k, dict(n=0, accept=0, refuse=0, abstain=0, accept_2tau=0, drops=[]))
        s["n"] += 1; s[r["preserve_tau"]["decision"]] += 1
        s["accept_2tau"] += r["preserve_2tau"]["decision"] == "accept"
        s["drops"].append(r["iou_before"] - r["iou_after"])
    out = dict(tau=tau, tau_source=f"max fitted-IoU drop over {len(app)} appearance-only edits (E/F/K)",
               records=recs, by_edit={f"{a}|{b}": v for (a, b), v in summ.items()})
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1)
    L = ["# B2: observed-evidence policy applied to all demo edits", "",
         f"TAU = {tau:.3f} ({out['tau_source']}). Preserve mode refuses an edit whose fitted input-view IoU",
         "drops by more than TAU. Counterfactual mode would keep it, labelled. Hypotheses the KB does not",
         "specify are abstentions by protocol.", "",
         "| demo | edit | n | accept @TAU | refuse @TAU | abstain | accept @2TAU | median IoU drop |", "|---|---|---|---|---|---|---|---|"]
    import statistics
    for (a, b), v in sorted(summ.items()):
        L.append(f"| {a} | {b} | {v['n']} | {v['accept']} | {v['refuse']} | {v['abstain']} | {v['accept_2tau']} | "
                 f"{statistics.median(v['drops']):.3f} |")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
