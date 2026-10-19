"""Executable leakage guards (protocol Part D).

Every synthetic image carries its root source photos (sha1) through the provenance chain
source image -> segmentation -> 3D asset -> edit -> render. Only roots in the PARENT
partition of known classes may feed generation; anything else raises LeakageError.

    from src.provenance.leakage import Guard
    g = Guard()                       # loads data/real_splits/{real_splits.parquet,LOCK.json}, verifies the lock
    g.assert_generation_roots(shas)   # every root must be a 'parent' image of a known class
    g.assert_not_near_test(phashes)   # no generated/derived image within HAM bits of a test image
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LeakageError(AssertionError):
    pass


def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


class Guard:
    def __init__(self, splits=f"{ROOT}/data/real_splits/real_splits.parquet", lock=f"{ROOT}/data/real_splits/LOCK.json"):
        self.lock = json.load(open(lock))
        self.S = pd.read_parquet(splits, columns=["sha1", "class_id", "partition", "phash", "group"])
        test = sorted(self.S[self.S.partition == "test"].sha1.unique())
        h = hashlib.sha256("\n".join(test).encode()).hexdigest()
        if h != self.lock["test_sha1_sha256"]:
            raise LeakageError("locked test set changed since LOCK.json was written")
        self.part = self.S.drop_duplicates("sha1").set_index("sha1").partition
        self.open_classes = set(self.lock["open_set_classes"])
        tp = self.S[self.S.partition == "test"].drop_duplicates("sha1").phash.astype("uint64").values
        self.test_ph = np.asarray(tp, dtype=np.uint64)
        self.ham = int(self.lock["ham_threshold"])

    def partition_of(self, sha1):
        return self.part.get(sha1, "unknown")

    def assert_generation_roots(self, shas):
        bad = {s: self.partition_of(s) for s in shas if self.partition_of(s) != "parent"}
        if bad:
            raise LeakageError(f"{len(bad)} generation roots outside the parent pool: {dict(list(bad.items())[:5])}")

    def assert_not_near_test(self, phashes, chunk=512):
        ph = np.asarray(phashes, dtype=np.uint64)
        for i in range(0, len(ph), chunk):
            d = _popcount(ph[i:i + chunk, None] ^ self.test_ph[None, :])
            hit = np.flatnonzero(d.min(1) <= self.ham)
            if len(hit):
                raise LeakageError(f"{len(hit)} derived images within {self.ham} bits of a locked test image")

    def known_class(self, class_id):
        return class_id not in self.open_classes
