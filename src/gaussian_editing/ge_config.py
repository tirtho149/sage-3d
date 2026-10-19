"""Structured GaussianEditor (edit-n2n) run configs.

Prompts never pass through the command line or OmegaConf's CLI grammar. A complete YAML is generated per run
(base configs/edit-n2n.yaml + collaborator hyperparameters + prompt/seed/steps/paths), validated by re-loading
it exactly the way threestudio does (OmegaConf.load -> merge -> resolve), and the prompt must round-trip
byte-identically. After a run, verify_run() checks GaussianEditor's own configs/parsed.yaml.

Run in the `gedit` env (omegaconf 2.3).
  ge_config.py make  <out.yaml> <prompt_file> <seed> <max_steps> <gs_source> <data_source> <exp_root>
  ge_config.py check <cfg.yaml> [<cfg.yaml> ...]            # preflight, no GPU
  ge_config.py verify <trial_dir> <cfg.yaml>                # post-run: parsed.yaml prompt == intended prompt
"""
import os, sys, re, json, hashlib, unicodedata
from omegaconf import OmegaConf

GE = os.path.expanduser("~/Desktop/Research/external/GaussianEditor")
BASE = f"{GE}/configs/edit-n2n.yaml"
# collaborator's submit_ge_edit.sh hyperparameters (unchanged from run_ge.sh)
HP = {"system.max_densify_percent": 0.01, "system.anchor_weight_init_g0": 0.05, "system.anchor_weight_init": 0.1,
      "system.anchor_weight_multiplier": 1.3, "system.gs_lr_scaler": 5, "system.gs_final_lr_scaler": 5,
      "system.color_lr_scaler": 5, "system.opacity_lr_scaler": 2, "system.scaling_lr_scaler": 2, "system.rotation_lr_scaler": 2,
      "system.loss.lambda_anchor_color": 0, "system.loss.lambda_anchor_geo": 50, "system.loss.lambda_anchor_scale": 50,
      "system.loss.lambda_anchor_opacity": 50, "system.densify_from_iter": 100, "system.densify_until_iter": 1501,
      "system.densification_interval": 100}
CLIP_MAX = 77
# Arm K/H appearance-only 3D editing: only colour (f_dc) is optimised. get_expon_lr_func returns 0 when both lr ends
# are 0, constant-lr groups get lr 0; densify_until_iter=0 disables densification AND pruning.
FROZEN_GEOMETRY = {"system.gs_lr_scaler": 0.0, "system.gs_final_lr_scaler": 0.0, "system.scaling_lr_scaler": 0.0,
                   "system.rotation_lr_scaler": 0.0, "system.opacity_lr_scaler": 0.0, "system.densify_until_iter": 0}


def normalize_prompt(p: str) -> str:
    """Single line, NFC, collapsed whitespace. Newlines/tabs are not meaningful to the CLIP tokenizer."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", p)).strip()


def escape_interp(p: str) -> str:
    # threestudio calls OmegaConf.resolve(), which turns an escaped "\\${" back into a live interpolation
    # (verified in tests), so "${" cannot be delivered verbatim: reject instead of silently altering the prompt.
    if "${" in p: raise ValueError("prompt contains '${', which GaussianEditor's config loader cannot deliver verbatim")
    return p


def clip_len(p):
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder="tokenizer", local_files_only=True)  # the tokenizer GaussianEditor uses
        return len(tok(p)["input_ids"])
    except Exception:
        return None


def make(out, prompt, seed, max_steps, gs_source, data_source, exp_root, tag="edit", extra=None):
    p = normalize_prompt(prompt)
    if not p: raise ValueError("empty prompt")
    cfg = OmegaConf.load(BASE)
    for k, v in {**HP, **(extra or {})}.items(): OmegaConf.update(cfg, k, v, merge=True)
    OmegaConf.update(cfg, "system.prompt_processor.prompt", escape_interp(p))
    for k, v in dict(seed=int(seed), tag=tag, exp_root_dir=exp_root, **{"trainer.max_steps": int(max_steps),
                     "system.gs_source": gs_source, "data.source": data_source}).items():
        OmegaConf.update(cfg, k, v, merge=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    OmegaConf.save(cfg, out)
    meta = dict(prompt=p, prompt_sha1=hashlib.sha1(p.encode()).hexdigest(), clip_tokens=clip_len(p), seed=int(seed),
                max_steps=int(max_steps), gs_source=gs_source, data_source=data_source, exp_root=exp_root, tag=tag, extra=extra or {})
    json.dump(meta, open(out + ".meta.json", "w"), indent=1, ensure_ascii=False)
    check(out); return out


def check(path):
    """Preflight: load exactly as threestudio.load_config does (no CLI args), resolve, compare prompt."""
    meta = json.load(open(path + ".meta.json"))
    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_cli([]))
    OmegaConf.resolve(cfg)
    got = cfg.system.prompt_processor.prompt
    errs = []
    if not isinstance(got, str): errs.append(f"prompt parsed as {type(got).__name__}")
    elif got != meta["prompt"]: errs.append(f"prompt mismatch: {got!r} != {meta['prompt']!r}")
    if cfg.tag != meta["tag"]: errs.append("tag mismatch")
    for k in ("gs_source", "data_source"):
        if not os.path.exists(meta[k]): errs.append(f"missing {k}: {meta[k]}")
    if meta.get("clip_tokens") and meta["clip_tokens"] > CLIP_MAX: errs.append(f"prompt {meta['clip_tokens']} CLIP tokens > {CLIP_MAX} (would be truncated)")
    if errs: raise ValueError(f"{path}: " + "; ".join(errs))
    return True


def verify_run(trial_dir, path):
    meta = json.load(open(path + ".meta.json"))
    parsed = OmegaConf.load(os.path.join(trial_dir, "configs", "parsed.yaml"))
    got = parsed.system.prompt_processor.prompt
    return dict(prompt_delivered=(got == meta["prompt"]), parsed_prompt=got,
                last_ply=os.path.exists(os.path.join(trial_dir, "save", "last.ply")))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        out, pf, seed, steps, gs, ds, root = sys.argv[2:9]
        extra = json.loads(sys.argv[9]) if len(sys.argv) > 9 else None       # e.g. FROZEN_GEOMETRY overrides
        print(make(out, open(pf, encoding="utf-8").read(), seed, steps, gs, ds, root, extra=extra))
    elif cmd == "check":
        for p in sys.argv[2:]: check(p); print("OK", p)
    elif cmd == "verify":
        r = verify_run(sys.argv[2], sys.argv[3]); print(json.dumps(r, ensure_ascii=False)); sys.exit(0 if r["prompt_delivered"] and r["last_ply"] else 3)
