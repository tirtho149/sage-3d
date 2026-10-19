"""Shared constants for the minimal downstream falsification test (reports/14_downstream_minimal.md)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/downstream_minimal"
SHARDS = Path.home() / "Desktop/Research/data/raw/SAGE/data"

# crop -> (pilot class, 3 largest other same-crop classes; Citrus has only 2 with >= 30 images)
CROPS = {
    "Peach": ("Leaf Curl", ["Bacterial Leaf Spot", "Brown Rot", "Peach Scab"]),
    "Corn": ("Smut", ["Common Rust", "Northern Leaf Blight", "Gray Leaf Spot"]),
    "Tomato": ("Leaf Mold", ["Tomato Leaf Curl Virus", "Septoria Leaf Blotch", "Bacterial Leaf Spot"]),
    "Citrus": ("Canker", ["Greening Disease", "Penicillium Fungi"]),
    "Mango": ("Anthracnose", ["Bacterial Canker", "Sooty Mould", "Die Back"]),
}
PILOTS = {
    "Peach": ["Peach__Leaf_Curl__0006", "Peach__Leaf_Curl__0063", "Peach__Leaf_Curl__0087"],
    "Corn": ["Corn__Smut__0012", "Corn__Smut__0025", "Corn__Smut__0052"],
    "Tomato": ["Tomato__Leaf_Mold__0015", "Tomato__Leaf_Mold__0052", "Tomato__Leaf_Mold__0092"],
    "Citrus": ["Citrus__Canker__0004", "Citrus__Canker__0012", "Citrus__Canker__0016"],
    "Mango": ["Mango__Anthracnose__0044", "Mango__Anthracnose__0049", "Mango__Anthracnose__0094"],
}
N_EVAL_CAP, N_TRAIN_CAP, N_BG = 200, 300, 20
VIEWS = [0, 3, 6, 9, 12, 15]
# render condition -> arms tried in order (first available is used; pre-registered fallbacks)
SYN_SOURCES = {"D": ["D_render"], "GEN3D": ["G", "E"], "KB3D": ["H", "K"], "J": ["J"], "F": ["F"]}
KS = [1, 4, 16]
N_DRAWS = 50
N_BOOT = 2000
PRIMARY_CONDS = ["R", "S", "D", "KB2D", "GEN3D", "KB3D"]
EXPLORATORY_CONDS = ["J", "F"]


def sid_to_specimen(sid):
    """'Peach__Leaf_Curl__0006' -> 'Peach__Leaf_Curl/0006' (specimen_manifest id)."""
    head, num = sid.rsplit("__", 1)
    return f"{head}/{num}"
