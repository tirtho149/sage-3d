# 03 — Knowledge Protocol (SAGE KB → structured, provenance-checked edit instructions)

Code: `src/kb/{schema,extract,retriever,validator,prompt_builder}.py`. Tests: `tests/test_kb.py` (5 tests, passing). Pilot export: `data/kb/pilot_kb.json`.

## 1. Source

Each SAGE (crop, disease) class carries:

- `visual_symptoms`: an extracted description.
- `symptom_quote`: a verbatim web quote.
- `symptom_source`: the URL.
- `pathogen`, `disease_type`, `plant_organ`.

Per the SAGE paper, expert agreement on sampled claims is 69–91%. This project uses those fields **as-is**. It does not re-crawl the sources, and adds no knowledge from LLMs or from memory.

## 2. Structured record (`schema.py`)

- **`KBRecord`:** crop, disease, canonical name, organ list, pathogen, type, the three text fields, image count, and a list of `SymptomSpec`s.
- **`SymptomSpec`:**
  - **`feature`:** colour, lesion, surface_growth, deformation, outgrowth, relief, or architecture.
  - **`organ`, `surface`:** surface is upper, lower, both, or unspecified.
  - **`attributes`:** matched terms, colours, direction.
  - **`evidence`:** see §3.
  - **`hedged`:** whether the source qualifies the claim.
  - **`provenance`:** URL, exact span, the field the span came from, and `verbatim` (true if the span occurs in the web quote itself).
- **Geometric features:** deformation, outgrowth, relief, and architecture are marked geometric. These are the claims a 2D edit cannot fully represent.

## 3. Evidence states (per specimen, never per disease)

| State | Meaning | Who can set it |
|---|---|---|
| `observed` | Visible in this specimen's input image | **Only** an annotator (validator V2 refuses otherwise) |
| `source_supported_unobserved` | Source says the disease shows it; not confirmed for this specimen | Default for unhedged KB claims |
| `uncertain` | Source hedges ("may", "sometimes", "in severe cases") | Automatic, from hedge lexicon |
| `unsupported` | No verbatim span backs it | Validator V1. Never edited in. |

A source-supported possibility is never converted into a confirmed symptom of an individual specimen.

## 4. Extraction (`extract.py`)

Extraction is rule-based on purpose: a fixed, versioned lexicon over clause-split source text, with no LLM, so it cannot invent content. Every spec is anchored to an exact clause. Recall is limited to the lexicon. The pilot classes yield 4–11 validated specs each.

## 5. Validation (`validator.py`)

- **V1:** the span must occur verbatim in the named field.
- **V2:** `observed` requires an annotation.
- **V3:** unsupported specs are never passed on.
- **V4:** every edit-plan operation must reference a validated spec.
- **V5:** the target organ must be in the KB organ list.

The `plant_organ` field is parsed both as a comma list and as a stringified Python list; both forms occur in SAGE.

## 6. Instructions for every arm (`prompt_builder.py`)

| Condition | Content | Used by arms |
|---|---|---|
| `generic` | "add disease symptoms to the {organ}" | B, E |
| `kb` | Up to 2 cited source clauses (≤22 words each, geometric first, unhedged first, near-duplicates removed) | C, F |
| `kb_no_geometry` | `kb` minus any clause containing a geometric term | ablation |
| `kb_mismatched` | `kb` of another disease of the same crop with **different** KB text (Jaccard ≤ 0.3) | negative control |

- **Same text for 2D and 3D.** The 2D and 3D arms receive byte-identical text.
- **Token limit:** all pilot instructions are ≤ 73 CLIP tokens. The CLIP text encoder used by InstructPix2Pix and GaussianEditor truncates at 77, which is asserted in the tests.
- **Empty conditions** are recorded as N/A with a reason, never silently replaced. Two cases:
  - `kb_no_geometry` for Corn Smut and Citrus Canker: every cited fruit clause is geometric.
  - Grape Leafroll (deferred class): every cited clause is geometric.

## 7. Pilot instructions (from `data/kb/pilot_kb.json`)

| Class | `kb` instruction (verbatim source clauses) | Mismatched control |
|---|---|---|
| Peach Leaf Curl | "…developing leaves become severely distorted (thickened and puckered), and have a reddish or purple cast; later, as spores form on the leaf surface, the leaves become powdery gray in color" | Bacterial Leaf Spot |
| Tomato Leaf Mold | "…olive-green to brown velvety mold forms on the lower leaf surface below leaf spots; pale greenish-yellow spots… form on the upper sides of leaves" | Septoria Leaf Blotch |
| Corn Smut | "…tumor-like galls that vary in size from less than 1 cm to more than 30 cm; young galls are white, smooth and firm…" | Northern Leaf Blight |
| Citrus Canker | "…brown spots that are oily in appearance and surrounded by a yellow halo, becoming corky with a crater-like depression" | Greening Disease (organ fallback: leaf clauses) |
| Mango Anthracnose | "…oval or irregular brown to deep brown sunken point-sized spots of varying sizes spread all over the leaf…" | Bacterial Canker |

## 8. KB data-quality findings

1. **Duplicated KB text.** Three Citrus classes carry text identical to Citrus Canker (word-Jaccard 1.0): *Alternaria Black Molds Stem Cankers*, *Botryosphaeria Rots And Cankers*, *Phytophthora Basal Canker*. The first was initially chosen as the mismatched control, which would have made the "negative" control identical to the positive. It is now excluded by the Jaccard rule. This needs a fix upstream in SAGE.
2. **Stale metadata file.** The HF `sage_metadata.parquet` is an older KB fill (236 classes vs 635 in the shards). Use the shards.
3. **Truncated source text.** Some quotes are cut mid-sentence (e.g. Corn Smut "…30 cm (0"). This is inherited from SAGE extraction.
4. **Organ field not per image.** `plant_organ` is a disease-level list. Per-image organ (e.g. the fruit vs leaf images of Citrus Canker) is set per specimen in `pilot.yaml` (`organ_override`).

## 9. Not yet implemented

- Severity grading from KB text.
- The `kb_no_organ` and `kb_with_uncertainty` ablations.
- Annotator interface for setting `observed`. Annotation is currently blocked: no raters.
