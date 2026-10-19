"""GaussianEditor arms for one specimen, sequential on one GPU (CUDA_VISIBLE_DEVICES set by caller).
  F  original GE + original pilot KB prompt (mixed stages; kept for comparability with the interim report)
  K  GE appearance-only (FROZEN_GEOMETRY) + structured KB appearance-only prompt, on the unedited asset
  H  geometry (J/seed<k>) then GE appearance-only + KB appearance-only prompt (method G4: geometry -> appearance)
Every run: generated+validated config, run_ge.sh (exit 0 only if prompt delivered and asset saved), then rig renders,
view metrics, Gaussian diff and KB attribute scores vs D. Resumable; results/phaseA/<sid>/<arm>/seed<k>/.
usage (any python with json): run_ge_arms.py <sid> <class_key> <arms e.g. F,K,H> [seed=0] [steps=200]
"""
import os, sys, json, subprocess, time, shutil
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.expanduser("~"); GEPY = f"{H}/micromamba/envs/gedit/bin/python"; SPY = f"{H}/micromamba/envs/sam3d/bin/python"
sid, key, arms = sys.argv[1], sys.argv[2], sys.argv[3].split(","); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
OUT = f"{R}/results/phaseA/{sid}"; SRC = f"{R}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"; D = f"{R}/results/pilot/{sid}/D"
S = json.load(open(f"{R}/data/kb/structured_specs.json"))[key]; P0 = json.load(open(f"{R}/results/pilot/{sid}/prompts.json"))["prompts"]
sys.path.insert(0, f"{R}/src/gaussian_editing"); FROZEN = {"system.gs_lr_scaler": 0.0, "system.gs_final_lr_scaler": 0.0, "system.scaling_lr_scaler": 0.0,
                   "system.rotation_lr_scaler": 0.0, "system.opacity_lr_scaler": 0.0, "system.densify_until_iter": 0}
GE_ENV = dict(os.environ, LD_LIBRARY_PATH=f"{H}/micromamba/envs/gedit/lib")
LOG = open(f"{OUT}/ge_runs.jsonl", "a")
# appearance pass prompt: KB appearance-only phrases; if the stage has none that can be split from geometry without
# paraphrase (Corn, Mango, Citrus fruit), fall back to the stage-consistent joint prompt (GE cannot change geometry when
# frozen, so only its appearance content can act). The choice is logged per run.
APP = S["prompts"]["kb_appearance_only"]["text"] or S["prompts"]["kb_joint_stage"]["text"]
APP_SRC = "kb_appearance_only" if S["prompts"]["kb_appearance_only"]["text"] else "kb_joint_stage (fallback)"


def sh(cmd, env=None, log=None):
    t = time.time(); r = subprocess.run(cmd, shell=True, env=env, stdout=open(log or os.devnull, "a"), stderr=subprocess.STDOUT); return r.returncode, time.time() - t


def claim(d):
    """Atomic per-run claim so several GPU workers can share one job list. A claim whose PID is dead is taken over."""
    c = f"{d}/.claim"
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        try:
            os.kill(int(open(c).read().strip() or 0), 0); return False        # owner alive -> skip
        except (ProcessLookupError, ValueError):
            open(c, "w").write(str(os.getpid())); return True                # stale -> take over


for arm in arms:
    d = f"{OUT}/{arm}/seed{seed}"; os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/asset.ply"): continue
    if not claim(d): print(f"skip {sid} {arm}: claimed by another worker", flush=True); continue
    if arm == "F":   prompt, src, views, extra = P0["kb"], SRC, D, None
    elif arm == "K": prompt, src, views, extra = APP, SRC, D, FROZEN
    elif arm == "H":
        if not os.path.exists(f"{OUT}/J/seed{seed}/asset.ply"):
            LOG.write(json.dumps(dict(arm=arm, seed=seed, status="N/A", reason="no J asset (KB has no geometric operator) -> H == K")) + "\n"); LOG.flush(); continue
        prompt, src, extra = APP, f"{OUT}/J/seed{seed}/asset.ply", FROZEN
        views = f"{d}/views_J"
        if not os.path.exists(f"{views}/rig.json"):
            sh(f"{GEPY} {R}/src/rendering/views.py {src} {views} --rig {D}/rig.json", GE_ENV, f"{d}/log.txt")
    else: raise ValueError(arm)
    if prompt is None:
        LOG.write(json.dumps(dict(arm=arm, seed=seed, status="N/A", reason="no KB prompt for this condition")) + "\n"); LOG.flush(); continue
    tmp = f"{d}/_ge"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    open(f"{tmp}/prompt.txt", "w").write(prompt)
    rc0, _ = sh(f"{GEPY} {R}/src/gaussian_editing/ge_config.py make {tmp}/cfg.yaml {tmp}/prompt.txt {seed} {steps} {src} {views} {tmp} '{json.dumps(extra or {})}'", GE_ENV, f"{d}/log.txt")
    rc, sec = sh(f"{R}/src/gaussian_editing/run_ge.sh {tmp}/cfg.yaml", None, f"{d}/log.txt") if rc0 == 0 else (2, 0)
    trial = sorted([os.path.join(r, x) for r, ds, _ in os.walk(f"{tmp}/edit-n2n") for x in ds if x.startswith("edit@")])
    ver = json.load(open(f"{trial[-1]}/verify.json")) if trial and os.path.exists(f"{trial[-1]}/verify.json") else {}
    rec = dict(arm=arm, seed=seed, steps=steps, prompt=prompt, prompt_source=("pilot kb (mixed stages)" if arm == "F" else APP_SRC), frozen_geometry=bool(extra), rc=rc, sec=round(sec, 1), verify=ver,
               t=time.strftime("%Y-%m-%d %H:%M:%S"))
    if rc == 0 and ver.get("prompt_delivered") and ver.get("last_ply"):
        shutil.copy(f"{trial[-1]}/save/last.ply", f"{d}/asset.ply")
        for r_, _, fs in os.walk(tmp):
            for f in fs:
                if f.endswith(".ckpt") or f == "last.ply": os.remove(os.path.join(r_, f))
        sh(f"{SPY} {R}/src/rendering/rig_render.py {d}/asset.ply {D}/rig.json {d}/render", None, f"{d}/log.txt")
        sh(f"{SPY} {R}/src/metrics/view_metrics.py {OUT}/D_render {d}/render {D}/rig.json {d}/view_metrics.json", None, f"{d}/log.txt")
        sh(f"{SPY} {R}/src/metrics/gaussian_diff.py {SRC} {d}/asset.ply {d}/gaussian_diff.json", None, f"{d}/log.txt")
        sh(f"{SPY} {R}/src/metrics/geom_attributes.py {SRC} {d}/asset.ply {d}/geom_attributes.json", None, f"{d}/log.txt")
        sh(f"{SPY} {R}/src/metrics/appearance_kb.py {OUT}/D_render {d}/render {D}/rig.json {key} {d}/appearance_kb.json", None, f"{d}/log.txt")
        rec["status"] = "OK"
    else:
        rec["status"] = "FAILED"
    LOG.write(json.dumps(rec) + "\n"); LOG.flush(); print(json.dumps({k: rec[k] for k in ("arm", "status", "rc", "sec")}), flush=True)
