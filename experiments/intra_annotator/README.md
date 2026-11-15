# Intra-annotator agreement: how consistent are my own hand labels?

**Question.** The test set will be labelled by hand. Hand labels are not perfect, so two questions follow:
1. **How high can a model score at best?** If I label the same frame twice, my two masks don't match exactly.
   A model can't be meaningfully judged above that level of agreement (the **IoU ceiling**).
2. **What tolerance τ should NSD use?** NSD (normalised surface distance) counts a predicted boundary as correct
   if it lies within τ pixels of the ground-truth boundary. τ should match how much the boundary moves when
   the *same* object is outlined twice, not be picked arbitrarily.

**Method.** Label the same 30 train frames several times in CVAT, with breaks in between and fixed labelling rules,
without looking at earlier runs. Then compare the runs pairwise with `compare_runs.py`.

## What the script measures
It keeps two kinds of disagreement apart:

| | What it is | Used for |
|---|---|---|
| **Whole frame** | IoU and NSD on the binary tool mask per frame. A tool labelled in only one run counts fully as disagreement. | IoU ceiling |
| **Matched objects** | Tool regions are matched across runs; for the same object outlined twice, the distance between the two outlines is measured. | τ for NSD |
| **Unmatched objects** | Tools found in one run only, listed with frame and size. | Detection consistency |

Details:
- **Masks, not polygons.** Comparisons use the exported pixel masks, so one tool drawn as one polygon or as
  two touching polygons gives the same result.
- **Matching.** Each run's mask is split into connected regions (8-connected); regions smaller than
  `--min-area` (default 50 px, stray clicks) are dropped and counted. Regions that overlap across runs are grouped,
  so a tool split into two regions in one run still matches the single region in the other.
- **Boundary distance.** For every boundary pixel of one outline, the distance to the nearest boundary pixel of the
  other, in both directions. The **95th percentile** over all matched objects is the suggested τ: it covers typical
  disagreement, not only the average pixel. The image edge is not treated as a boundary, since a tool leaving the
  frame is cut by the camera, not outlined.
- **Ignored pixels.** Pixels labelled `ignore` in either run are left out of every number.
- **Empty frames.** Frames with no tool in either run have no boundary and are left out (and counted).

## Usage
Input: two CVAT exports in the **Segmentation mask 1.1** format (the folder holding `labelmap.txt`).
```bash
python experiments/intra_annotator/compare_runs.py RUN_A_MASKS RUN_B_MASKS --out OUT_DIR [--min-area 50]
```
Writes `summary.md`, `per_frame.csv`, `per_object.csv` and `boundary_distances.png` to `OUT_DIR`.
The annotations themselves are not in this repo.

## Biases and limits
- **Memory.** Same-day repeats partly remember earlier outlines, so agreement looks better than it is (τ too small).
  At least one repeat should be days later.
- **Learning.** Run 1 was my first labelling attempt, so it's treated as a warm-up; later runs are the main comparison.
- **Touching tools merge.** Matching on connected regions can't separate two tools that touch. If only one run labels
  the second tool, it counts as a matched object with a large boundary distance instead of a missed tool, which
  inflates τ. Matching on CVAT polygon instances would fix this.
- **Resolution.** τ is in pixels at native frame resolution (widths 828–1280 px here). Rescale it if the model works
  at another resolution.
- **One annotator.** This measures my consistency with myself, a lower bound on disagreement between different
  people (inter-annotator).

## First result (run 1 vs run 2, 30 frames, same day)
- Whole-frame IoU **0.875** (mean over 29 frames; 1 frame empty in both).
- 3 tools found in one run only.
- Matched objects (45): IoU **0.883**; boundary distance median 2.8 px, **p95 14.8 px**.

The p95 is inflated by the two effects above (a merged pair of touching tools, a tool end stopping short of the
image edge), so ~15 px is an upper estimate. The typical outline difference is ~3–5 px.
