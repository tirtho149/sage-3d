"""KB provenance tests. Uses real SAGE KB records; the tampered spec is a synthetic fixture."""
import sys, json, copy
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/src")
from kb.retriever import get_record
from kb.validator import validate, check_plan
from kb.schema import Evidence


def test_every_spec_is_verbatim_backed():
    r = get_record("Peach", "Leaf Curl"); v = validate(r)
    texts = {"symptom_quote": r.symptom_quote, "visual_symptoms": r.visual_symptoms}
    assert v.passed and all(s.provenance.span in texts[s.provenance.field_name] for s in v.passed)


def test_fabricated_span_rejected():
    r = get_record("Peach", "Leaf Curl"); bad = copy.deepcopy(r.specs[0])
    bad.provenance.span = "leaves develop bright blue stripes"   # SYNTHETIC: not in source
    r.specs.append(bad); v = validate(r)
    assert any(reason.startswith("V1") for _, reason in v.rejected)


def test_observed_requires_annotation():
    r = get_record("Tomato", "Leaf Mold"); v = validate(r)
    assert all(s.evidence != Evidence.OBSERVED for s in v.passed)
    v2 = validate(get_record("Tomato", "Leaf Mold"), observed_annotations={0: True})
    assert any(s.evidence == Evidence.OBSERVED for s in v2.passed)


def test_unbacked_plan_op_flagged():
    v = validate(get_record("Corn", "Smut"))
    assert check_plan([{"op": "add_gall", "spec_index": 0}], v) == []
    assert check_plan([{"op": "add_gall"}], v) == ["op0:add_gall:unbacked"]


def test_pilot_prompts_fit_clip():
    d = json.load(open(__file__.rsplit("/tests/", 1)[0] + "/data/kb/pilot_kb.json"))["classes"]
    for k, c in d.items():
        for cond, p in c["prompts"].items():
            assert p["clip_tokens"] <= 77, (k, cond, p["clip_tokens"])
