# VisionBoard architecture

```text
Browser: OpenSeadragon + canvas overlays
    │ JSON, DZI descriptors, PNG tiles
FastAPI
    ├── store.py        source and scene discovery
    ├── annotations.py prediction/GT/vector/detail parsing and delete filtering
    ├── evaluate.py    polygon-IoU TP/FP/FN matching
    ├── confusion.py   Test85 parent/child confusion events and deep-link records
    ├── tiler.py       lossless Deep Zoom generation with baked Delete masks
    ├── jobs.py        serial inference jobs
    └── adapters/      optional model subprocess integration
```

## Image delivery

The server builds a Deep Zoom pyramid on first view and caches it under `data/tile_cache/`.
The browser downloads only tiles visible in the current viewport. Image pixels are never
resized by the GUI before analysis; the pyramid is a viewing mechanism only. Delete polygons
are filled black in the generated tiles, never in the source image, so ignored content cannot
appear transiently during zoom.

## Overlay rendering

Boxes are rendered on a transparent canvas over OpenSeadragon. Image coordinates are
converted to viewer coordinates on every viewport update, so annotations pan and zoom with
the image and keep a constant screen-space stroke width.

Stage 1, Stage 2, GT, Stage0, and Grad-CAM visibility is controlled entirely on the frontend;
metrics are requested independently for both stages. The global overlay control affects the
canvas only, so baked Delete blackout remains permanent.

## Evaluation

The backend filters confidence, applies the selected Stage 2 gating mode, flags predictions
and GT inside delete regions, and performs greedy polygon-IoU matching. Three results are requested
for every open scene:

1. Stage 1 class-agnostic detection;
2. Stage 2 class-agnostic detection;
3. Stage 2 class-aware classification.

Layer checkboxes affect visualization, not these metric panels. Evaluation coloring uses the
result associated with the active visible layer.

Class-aware evaluation also returns a non-metric `error_matches` map. It associates a
wrong-class FP with the overlapping FN at the same IoU threshold, enabling the `Q` shortcut
without changing TP/FP/FN counts.

## Confusion page

`/confusion` requests dataset-wide Stage 2 events from `/api/confusion`. The backend exposes
both labels_simple P_Camo handling and actual-parent handling. Matrix records carry scene,
instance, and prediction identifiers used to deep-link into `/`; the return URL preserves the
matrix filters and view.

## Raw thresholds and vectors

Raw Stage 2 candidates are gated by either one global threshold or the shipped per-class
thresholds. Priority-aware cross-class dedup follows gating. The optional vector file is then
matched spatially to surviving predictions to expose top-three class scores. Vector scores
never change metrics.

## Runtime state

The application creates these untracked paths as needed:

```text
data/cache/
data/tile_cache/
data/jobs/
data/generated_preds/
data/uploads/
data/sources.json
```

No database is required.

