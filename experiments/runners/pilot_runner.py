"""Pilot orchestrator (one GPU). Per specimen, after SAM 3D seed-0 splat exists:

  E1   input-view render + fidelity metrics                    (sam3d env)
  D    18-view rig renders of the unedited asset               (gedit env)
  B/C  2D IP2P edits of the input image: generic / KB, seeds 0-2;
       + KB-no-geometry and KB-mismatched (seed 0) ablations   (.venv)
  Bmv/Cmv  independent per-view 2D edits of the 8 ring views (for E4)
  E/F  GaussianEditor generic / KB, seed 0, 200 steps, 18 views (gedit env)
       + 18-view renders of the edited assets

Everything is resumable (skips finished outputs) and logs runtime per step.
usage: pilot_runner.py <gpu> <specimen substrings comma-sep | all>
"""
import os, sys, json, time, subprocess, shutil, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOME = os.path.expanduser("~")
GPU, SEL = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(f"{ROOT}/experiments/configs/pilot.yaml"))
kb = json.load(open(f"{ROOT}/data/kb/pilot_kb.json"))["classes"]
SAM3D_PY = f"{HOME}/micromamba/envs/sam3d/bin/python"; GE_PY = f"{HOME}/micromamba/envs/gedit/bin/python"
VENV_PY = f"{HOME}/Desktop/Research/.venv/bin/python"
GE_STEPS, GE_NAZ, RING = 200, 8, list(range(8))      # 8 azimuths x (+20,-20) + top + bottom = 18 views; ring = +20 deg views
OUT = f"{ROOT}/results/pilot"
LOG = open(f"{OUT}/runner_gpu{GPU}.jsonl", "a")


def sh(cmd, env_extra=None, log=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, **(env_extra or {}))
    t = time.time()
    with open(log or os.devnull, "a") as f:
        r = subprocess.run(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, round(time.time() - t, 1)


def record(**k):
    LOG.write(json.dumps(dict(k, t=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n"); LOG.flush()


specs = [s for s in cfg["specimens"] if SEL == "all" or any(x in s for x in SEL.split(","))]
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)   # set by pilot_chain.sh for SAM 3D; torch 2.0.1 (gedit) rejects expandable_segments
GE_ENV = dict(LD_LIBRARY_PATH=f"{HOME}/micromamba/envs/gedit/lib")
SAM_ENV = dict(CONDA_PREFIX=f"{HOME}/micromamba/envs/sam3d", CUDA_HOME=f"{HOME}/micromamba/envs/sam3d", LIDRA_SKIP_INIT="true")
pending = list(specs)
while pending:
    ready = [s for s in pending if os.path.exists(f"{OUT}/sam3d/{s.replace('/', '__')}/seed0/splat.ply")]
    if not ready:
        time.sleep(60); continue
    s = ready[0]; pending.remove(s)
    key, sid = s.split("/")[0], s.replace("/", "__")
    os.makedirs(f"{OUT}/{sid}", exist_ok=True)
    try:
        fd = os.open(f"{OUT}/{sid}/.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, GPU.encode()); os.close(fd)
    except FileExistsError:
        if open(f"{OUT}/{sid}/.lock").read().strip() != GPU:
            continue    # another GPU's runner owns this specimen
    inp, d0 = f"{ROOT}/data/inputs/{sid}", f"{OUT}/sam3d/{sid}/seed0"
    P = kb[key]["prompts"]; organ = cfg.get("organ_override", {}).get(s, cfg["organ"][key])
    base = f"{OUT}/{sid}"; os.makedirs(base, exist_ok=True); lg = f"{base}/log.txt"
    # prompts: organ override (Citrus leaf specimen) -> rebuild organ word only for generic; KB text is class-level
    prompts = {c: (P[c]["text"] if c != "generic" else f"add disease symptoms to the {organ if not (key=='Corn__Smut') else 'ear'}") for c in P}
    json.dump(dict(specimen=s, organ=organ, prompts=prompts), open(f"{base}/prompts.json", "w"), indent=1)
    # E1
    if not os.path.exists(f"{d0}/e1.json"):
        rc, sec = sh(f"{SAM3D_PY} {ROOT}/src/rendering/input_view.py {d0} {inp}", SAM_ENV, lg); record(spec=s, step="E1", rc=rc, sec=sec)
    # D views
    if not os.path.exists(f"{base}/D/rig.json"):
        rc, sec = sh(f"{GE_PY} {ROOT}/src/rendering/views.py {d0}/splat.ply {base}/D --n_az {GE_NAZ}", GE_ENV, lg); record(spec=s, step="D_views", rc=rc, sec=sec)
    # 2D jobs
    jobs = []
    for cond, arm, seeds in [("generic", "B", [0, 1, 2]), ("kb", "C", [0, 1, 2]), ("kb_no_geometry", "C_nogeo", [0]), ("kb_mismatched", "C_mismatch", [0])]:
        if not prompts.get(cond):
            continue
        for k in seeds:
            jobs.append(dict(image=f"{inp}/image.png", mask=f"{inp}/0.png", prompt=prompts[cond], seed=k, out=f"{base}/{arm}/seed{k}.png"))
    for cond, arm in [("generic", "Bmv"), ("kb", "Cmv")]:
        for v in RING:
            jobs.append(dict(image=f"{base}/D/images/{v:03d}.png", mask=None, prompt=prompts[cond], seed=0, out=f"{base}/{arm}/{v:03d}.png"))
    jf = f"{base}/ip2p_jobs.json"; json.dump(jobs, open(jf, "w"), indent=1)
    rc, sec = sh(f"{VENV_PY} {ROOT}/src/editing2d/ip2p_batch.py {jf}", None, lg); record(spec=s, step="2D", rc=rc, sec=sec, n=len(jobs))
    # 3D edits
    for cond, arm in [("generic", "E"), ("kb", "F")]:
        od = f"{base}/{arm}/seed0"
        if os.path.exists(f"{od}/last.ply"):
            continue
        tmp = f"{base}/{arm}/_ge"; shutil.rmtree(tmp, ignore_errors=True)
        if not os.path.exists(f"{d0}/splat_ge.ply"):   # inf opacity logits -> NaN anchor loss -> every fp16 step skipped (audit F-GE-2)
            sh(f"{SAM3D_PY} {ROOT}/src/gaussian_editing/sanitize_splat.py {d0}/splat.ply {d0}/splat_ge.ply", None, lg)
        os.makedirs(tmp, exist_ok=True); open(f"{tmp}/prompt.txt", "w").write(prompts[cond])
        rc, sec = sh(f"{ROOT}/src/gaussian_editing/run_ge.sh {d0}/splat_ge.ply {base}/D \"$(cat {tmp}/prompt.txt)\" 0 {GE_STEPS} {tmp}", None, lg)
        ply = [os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f == "last.ply"]
        os.makedirs(od, exist_ok=True)
        if ply:
            shutil.copy(ply[0], f"{od}/last.ply")
            for r, _, fs in os.walk(tmp):  # disk: drop GE checkpoints + duplicate ply (configs/logs kept)
                for f in fs:
                    if f.endswith(".ckpt") or f == "last.ply": os.remove(os.path.join(r, f))
            rc2, sec2 = sh(f"{GE_PY} {ROOT}/src/rendering/views.py {od}/last.ply {od}/views --rig {base}/D/rig.json", GE_ENV, lg)
        record(spec=s, step=f"GE_{arm}", rc=rc, sec=sec, ok=bool(ply), prompt=prompts[cond])
    record(spec=s, step="done")
print("ALL DONE", flush=True)
