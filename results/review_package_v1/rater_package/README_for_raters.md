# Plant-disease image review: instructions for raters

Package: `review_package_v1_bedf5fbd3b`  
Number of items: 84  
Estimated time: about 57 minutes (about 35 seconds per item after the first few, plus about 8 minutes to read these instructions). You may pause and resume.

## What you are asked to do

Each item shows a REAL photograph of a plant organ (green label) and a *candidate* image of the same specimen (grey label). A candidate is either a single image or several views of the same organ from different directions (front, back face, side, from above, from below). Candidates were produced in different ways; some may be unaltered. You are not told how any candidate was produced, and you should not try to guess.

The REAL photo may already show symptoms of the disease; this is expected. Judge whether the candidate is a realistic example of the described disease at the stated stage, whatever the REAL photo shows.

For each candidate, judge it against the written disease description shown on the item page (quoted verbatim from an extension or teaching source) for the stated stage. Judge the candidate, not the REAL photo; the REAL photo is there so you can see the specimen before any change.

Some items appear more than once. Rate each appearance independently; do not try to remember earlier scores.

## How to rate

1. Open `index.html` in a web browser (Chrome, Firefox, Edge or Safari). No internet connection is needed; nothing is sent anywhere.
2. Enter your rater code (given to you by the study coordinator; do not use your name).
3. For each item, choose a score from 1 to 5 for each of the six criteria (or NA if the criterion cannot be judged from the images), answer the yes/no question, and add a comment if you wish.
4. Click the image to open it at full size.
5. Your answers are saved in this browser as you go. When finished, click **Export CSV** and send the downloaded file to the study coordinator. You can also export part-way through as a backup.
6. If the HTML form does not work, fill in `rating_sheet.csv` directly (one row per item; item images are in `items/`).

## Criteria

### (a) Symptom presence matches the description

Are the symptoms named in the disease description for this stage present in the candidate, and absent symptoms not invented?

- **1**: No symptom of the described disease is visible; the organ looks healthy, or shows only a different disease.
- **2**: One weak or doubtful feature resembles the description; most named symptoms are missing or a different disease dominates.
- **3**: Some named symptoms are present and recognisable, but at least one key symptom is missing or clearly wrong in colour or texture.
- **4**: All key named symptoms are present and recognisable; minor differences in colour, extent or texture from the description.
- **5**: All named symptoms are present and match the description in colour, texture and extent; nothing would be added or removed.

### (b) Location and distribution plausible

Is the location and spatial distribution of the symptoms plausible for this disease on this organ (e.g. galls replacing or arising from kernels on the ear; olive-brown sporulation on the lower surface beneath upper-surface yellow spots; puckering and thickening spread over the lamina)?

- **1**: Symptoms are on the wrong organ or wrong part of the organ (e.g. galls floating off the ear, spots only on the background), or the pattern is impossible for this disease.
- **2**: Mostly implausible: symptoms in a pattern that a pathologist would not expect (regular grid, uniform tint over the whole image, only on the margin of the image).
- **3**: Partly plausible: correct organ, but density, clustering or position is unusual for this disease (e.g. sporulation and spots not co-located).
- **4**: Plausible distribution with a minor oddity (e.g. slightly too uniform, slightly too sparse).
- **5**: Distribution as seen in field specimens of this disease and organ.

### (c) Shape / geometry plausible

Is the 3D form of the organ plausible for this disease and stage (curl direction and amplitude, blister or pucker form, gall volume and attachment, lesion depression)? For a flat 2D image judge only what can be seen.

- **1**: Shape is impossible or clearly broken: torn, folded through itself, spikes, detached fragments, or a shape change the disease does not cause.
- **2**: Shape change present but of the wrong type (e.g. smooth uniform bending where puckering is expected; galls as flat discs) or grossly wrong magnitude.
- **3**: Shape change of the correct type but unconvincing in form or magnitude, or no shape change where the description implies one.
- **4**: Correct type and plausible magnitude; minor irregularities.
- **5**: Form indistinguishable from real specimens of this disease and stage; or, where the description implies no shape change, the organ shape is unaltered and natural.

### (d) Surface correctness (upper vs lower)

Are symptoms on the correct surface (upper vs lower) and consistent between views of the same organ? Use NA if the description names no surface and the question cannot apply.

- **1**: Symptoms are on the wrong surface (e.g. velvety mold on the upper surface only), or the two surfaces show contradictory symptoms.
- **2**: Mostly on the wrong surface; or symptoms bleed through identically to both surfaces where the disease does not do this.
- **3**: Mixed: correct surface carries some symptoms, but the opposite surface also carries features it should not.
- **4**: Correct surface; minor bleed-through or inconsistency between views.
- **5**: Each surface carries exactly the symptoms the description assigns to it, consistently across views.

### (e) Absence of non-biological artefacts

Are there artefacts that a pathologist would recognise as not biological (texture grain or noise, blotchy colour drift, view-to-view flicker, haze or floaters around the organ, blurred or smeared regions, seams)? Higher score = fewer artefacts.

- **1**: Artefacts dominate; the organ could not be used as an example at all.
- **2**: Conspicuous artefacts over a large part of the organ.
- **3**: Noticeable artefacts in one region or one view; usable with caveats.
- **4**: Minor artefacts visible only on close inspection.
- **5**: No artefacts visible.

### (f) Overall: acceptable realistic example

Would you accept this candidate as a realistic example of this disease at the stated stage (for teaching, a field guide or training data)?

- **1**: Definitely not; misleading if used.
- **2**: Probably not; more wrong than right.
- **3**: Uncertain; would need correction before use.
- **4**: Probably yes, with minor reservations.
- **5**: Definitely yes; as good as a real photograph of this disease and stage.

### Contains a feature the description excludes for this stage (yes / no)

Does the candidate contain any feature listed under 'should not be present at this stage' on the item page, any symptom of a different disease, or a change to background or unaffected organs?

NA: Use NA only when the images do not allow the criterion to be judged (for example, the organ surface named in the criterion is not visible in any panel). NA is treated as missing data, not as a low score.

## Disease descriptions used in this package

### Peach leaf curl (leaf)

> Developing leaves become severely distorted (thickened and puckered), and have a reddish or purple cast. Later, as spores form on the leaf surface, the leaves become powdery gray in color. Shortly after this, the leaves turn yellow or brown and drop.

Source: https://ohioline.osu.edu/factsheet/plpath-fru-26

Stage to judge (stage 1 of 3): Developing leaves become severely distorted (thickened and puckered), and have a reddish or purple cast.

Should not be present at this stage (later or other stage): Later, as spores form on the leaf surface, the leaves become powdery gray in color. Shortly after this, the leaves turn yellow or brown and drop.

### Tomato leaf mold (leaf)

> Pale greenish-yellow spots, usually less than 1/4 inch, with no definite margins, form on the upper sides of leaves. Olive-green to brown velvety mold forms on the lower leaf surface below leaf spots. Leaves wither and die but often remain attached to the plant. Fruit infections start as a smooth black irregular area on the stem end of the fruit.

Source: https://extension.umn.edu/disease-management/tomato-leaf-mold

Stage to judge (stage 1 of 1): Pale greenish-yellow spots, usually less than 1/4 inch, with no definite margins, form on the upper sides of leaves. Olive-green to brown velvety mold forms on the lower leaf surface below leaf spots. Leaves wither and die but often remain attached to the plant. Fruit infections start as a smooth black irregular area on the stem end of the fruit.

### Corn (maize) common smut (ear)

> The most obvious symptoms are tumor-like galls that vary in size from less than 1 cm to more than 30 cm (0.4-12 inch) in diameter.

Source: https://www.apsnet.org/edcenter/pdlessons/Pages/CornSmut.aspx

Stage to judge (stage 1 of 1): The most obvious symptoms are tumor-like galls that vary in size from less than 1 cm to more than 30 cm (0.4-12 inch) in diameter.

## Questions

Contact the study coordinator. Please do not discuss items with other raters until all ratings have been returned.
