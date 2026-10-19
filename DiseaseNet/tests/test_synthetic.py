"""Synthetic dataset integrity (protocol Part S): provenance, background independence, balanced arms."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.provenance.leakage import Guard  # noqa: E402

ST = f"{ROOT}/data/synthetic/stage0"
MAN = [f"{ST}/{m}" for m in ("manifest_3d.csv", "manifest_2d.csv") if os.path.exists(f"{ST}/{m}")]
pytestmark = pytest.mark.skipif(not MAN, reason="no synthetic manifest yet")


@pytest.fixture(scope="module")
def syn():
    return pd.concat([pd.read_csv(m) for m in MAN])


def test_every_root_is_a_parent_image(syn):
    Guard().assert_generation_roots(syn.root_sha1.unique().tolist())


def test_backgrounds_are_class_independent(syn):
    cls = syn.drop_duplicates("asset_id").set_index("asset_id").class_id
    r = syn[syn.cond.isin(["R2", "R5"])]
    assert (r.bg_asset.map(cls) != r.class_id).all()


def test_no_duplicate_paths_and_files_exist(syn):
    assert not syn.path.duplicated().any()
    assert all(os.path.exists(os.path.join(ROOT, p)) for p in syn.path.sample(min(200, len(syn)), random_state=0))


def test_arms_have_equal_budget_per_asset(syn):
    n = syn.groupby(["arm", "asset_id"]).size()
    assert n.nunique() == 1, n.value_counts().to_dict()


def test_only_stage0_classes(syn):
    classes = {c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]}
    assert set(syn.class_id) <= classes
