"""Arm F prompt-safety tests (run in gedit env): prompts must survive config generation byte-identically."""
import os, sys, json, tempfile, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "gaussian_editing"))
import ge_config as G

CASES = {
    "colon": "make this peach leaf show leaf curl: leaves become distorted",
    "quotes": 'leaves show "olive-green" mold and it\'s velvety',
    "newline": "line one\nline two\ttabbed",
    "braces": "spots {small} and [large], (oily) ; 50% coverage",
    "unicode": "Xanthomonas citri ssp. citri – lésions liégeuses, 1 cm–30 cm, ½ size",
    "yaml_special": "- dash start: key: value # not a comment & anchor *alias !tag",
    "real_kb": "make this peach leaf show leaf curl: developing leaves become severely distorted (thickened and puckered), and have a reddish or purple cast; later, as spores form on the leaf surface, the leaves become powdery gray in color",
}


@pytest.mark.parametrize("name", list(CASES))
def test_roundtrip(name, tmp_path):
    gs = tmp_path / "a.ply"; gs.write_text("x"); ds = tmp_path / "views"; ds.mkdir()
    out = G.make(str(tmp_path / "cfg.yaml"), CASES[name], 0, 10, str(gs), str(ds), str(tmp_path / "exp"))
    meta = json.load(open(out + ".meta.json"))
    assert meta["prompt"] == G.normalize_prompt(CASES[name])
    assert G.check(out)


def test_interpolation_rejected(tmp_path):
    with pytest.raises(ValueError):
        G.make(str(tmp_path / "c.yaml"), "literal ${system.seed}", 0, 10, "a", "b", "c")


def test_empty_rejected(tmp_path):
    with pytest.raises(ValueError):
        G.make(str(tmp_path / "c.yaml"), "  \n ", 0, 10, "a", "b", "c")


def test_old_cli_path_breaks():
    """Documents the original bug: OmegaConf CLI grammar turns 'a: b' into a mapping."""
    from omegaconf import OmegaConf
    c = OmegaConf.from_cli([f"p={CASES['colon']}"])
    assert not isinstance(c.p, str)
