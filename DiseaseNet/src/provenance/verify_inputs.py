"""Run the leakage Guard over generation inputs (Research .venv) and write a verified list.

Environments without a parquet engine (the sam3d env) cannot load the Guard; they read this
file instead and refuse any input that is not listed with the current lock hash.
usage: ~/Desktop/Research/.venv/bin/python -m src.provenance.verify_inputs [stage]
"""
import glob
import json
import os
import sys

from src.provenance.leakage import Guard, ROOT


def main(stage="stage0"):
    g = Guard(); out = {}
    for d in sorted(glob.glob(f"{ROOT}/data/synthetic/{stage}/inputs/*")):
        roots = json.load(open(f"{d}/mask_meta.json"))["root_sha1"]
        g.assert_generation_roots(roots)
        out[os.path.basename(d)] = roots
    json.dump(dict(lock=g.lock["test_sha1_sha256"], inputs=out),
              open(f"{ROOT}/data/synthetic/{stage}/verified_inputs.json", "w"), indent=1)
    print(f"verified {len(out)} inputs against lock {g.lock['test_sha1_sha256'][:12]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
