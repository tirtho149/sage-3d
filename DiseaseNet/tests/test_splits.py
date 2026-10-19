"""Split / provenance integrity (protocol Part S: leakage, parent-asset split integrity)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.provenance.leakage import Guard, LeakageError  # noqa: E402

S = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet")
LOCK = json.load(open(f"{ROOT}/data/real_splits/LOCK.json"))


def test_lock_matches():
    Guard()                                   # raises if the locked test set changed


def test_groups_never_split_across_known_partitions():
    k = S[S.partition.isin(["parent", "adapt", "val", "test"])].drop_duplicates(["group", "partition"])
    assert k.group.duplicated().sum() == 0


def test_open_set_classes_have_no_known_partition():
    o = S[S.class_id.isin(LOCK["open_set_classes"])]
    assert set(o.partition) == {"openset"}


def test_open_set_groups_absent_from_known_training_pools():
    og = set(S[S.partition == "openset"].group)
    assert not S[S.partition.isin(["parent", "adapt"]) & S.group.isin(og)].shape[0]


def test_existing_asset_sources_in_parent():
    assert LOCK["asset_source_images_forced_parent"] == 15


def test_guard_rejects_test_roots_and_near_duplicates():
    g = Guard()
    t = S[S.partition == "test"].iloc[0]
    with pytest.raises(LeakageError):
        g.assert_generation_roots([t.sha1])
    with pytest.raises(LeakageError):
        g.assert_not_near_test([np.uint64(t.phash) ^ np.uint64(0b11)])     # 2 bits away
    p = S[S.partition == "parent"].iloc[0]
    g.assert_generation_roots([p.sha1])                                    # allowed


def test_every_known_class_has_parent_and_test():
    k = S[~S.class_id.isin(LOCK["open_set_classes"])].drop_duplicates(["class_id", "sha1"])
    c = k.groupby(["class_id", "partition"]).size().unstack(fill_value=0)
    assert (c["parent"] > 0).all() and (c["test"] > 0).all()
