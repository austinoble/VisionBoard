# VisionBoard

VisionBoard is the source-only release of the Lowres Detection GUI V8. It visualizes large
satellite scenes with Stage 1 object proposals, Stage 2 classified detections, oriented
ground truth, stable instance IDs, and per-scene evaluation metrics.

This repository intentionally contains **no imagery, labels, predictions, model weights,
tile caches, or inference outputs**. Supply the dataset locally or point the application to
an external dataset root.

## Features

- Deep-zoom viewing of full-resolution PNG/JPEG/TIFF scenes with lossless PNG tiles.
- Crisp HBB/OBB vector overlays at every zoom level.
- Independent Stage 1, Stage 2, and ground-truth visual layers.
- Stage 1 detection, Stage 2 detection, and Stage 2 classification metrics.
- Exact polygon-IoU evaluation against `labels_simple`.
- Shipped Stage 2 per-class thresholds enabled by default.
- Delete-region blackout and exclusion from boxes, arrows, and metrics.
- Search by two-character image ID or complete five-character instance ID.
- Per-class arrow navigation through TP/FP predictions and FN ground truth.
- Collapsible `labels_detailed` metadata and Stage 2 top-three class scores.
- Full-scene Stage0 objectness heatmaps, off by default, with opacity control.
- Selected-prediction Grad-CAM for Stage1 and the highest-confidence Stage2 ensemble
  contributor, off by default.
- Mutually exclusive Stage0/Grad-CAM explanations so their color maps cannot be confused.
- Sorted Previous/Next scene browsing without entering an image ID.
- Nearest-unvisited spatial instance navigation with deterministic history.
- A permanent two-row bottom control dock, leaving more horizontal space for imagery.

## Quick start

```bash
git clone <repository-url>
cd VisionBoard
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

export VISIONBOARD_DATA_ROOT=/absolute/path/to/lowres_full86
bash run_gui.sh          # port 7860
# bash run_gui.sh 8080   # alternate port
```

Open `http://localhost:7860`. Set `GUI_PYTHON=/path/to/python` to use an existing Python
environment. Interactive API documentation is available at `/api/docs`.

## Platform data layout

Every scene is joined by its filename stem:

```text
lowres_full86/
├── images/                         # displayed images2x PNGs
│   └── <scene>.png
├── predictions/
│   ├── stage1/<scene>.txt          # deployed normalized Stage 1 boxes
│   ├── stage2/<scene>.txt          # deployed normalized Stage 2 boxes
│   ├── stage2_raw/<scene>.txt      # raw pre-calibration Stage 2 candidates
│   └── stage2_vectors/<scene>.txt  # optional 8-class vectors for top-three display
├── classes/
│   ├── stage1.txt
│   └── stage2.txt
├── heatmaps/stage0/<scene>.npy     # native-grid float heat in [0,1]
├── explainability/gradcam/
│   ├── manifests/<scene>.json      # prediction ID -> CAM path, bounds, provenance
│   └── cams/{stage1,stage2}/*.png  # full-context color CAM crops
└── gt/
    ├── labels_simple/<scene>.txt   # evaluation ground truth
    ├── labels_delete/<scene>.txt   # ignored/blackout polygons
    └── labels_detail_id/<scene>.txt # 25 detailed columns + stable ID
```

The viewed image is the 2× super-resolved image. The current coordinate contract is:

- deployed Stage 1/2 coordinates are normalized to the displayed image;
- `labels_simple` is in displayed-image pixels (`gt_scale=1.0`);
- `labels_detail_id` is in native pixels and is scaled by 2 for display;
- raw Stage 2 and vector coordinates are absolute displayed-image pixels.

The default root is `data/lowres_full80` only when no external root is supplied. Prefer
environment variables for the 86-scene deployment:

```bash
export VISIONBOARD_DATA_ROOT=/datasets/lowres_full80
export VISIONBOARD_STAGE2_RAW=/runs/stage2/pred_all_c05
export VISIONBOARD_STAGE2_VECTORS=/runs/stage2/pred_vectors
export VISIONBOARD_STAGE0_ROOT=/runs/heatmaps/stage0
export VISIONBOARD_GRADCAM_ROOT=/runs/explainability/gradcam
export VISIONBOARD_SOURCE_NAME=lowres_full86
export VISIONBOARD_DATA_TITLE="Low-resolution evaluation set (86 scenes)"
```

The raw/vector defaults are folders below `VISIONBOARD_DATA_ROOT/predictions/`.

## V8 explanation generation

V8 precomputes explanations; the web server never loads model weights or occupies a GPU.
This keeps browsing responsive and makes the displayed CAM reproducible. Generate Stage2
first, then Stage1, binding the experiment to the intended physical GPU UUID:

```bash
export CUDA_VISIBLE_DEVICES=GPU-6f83fb5a-9b8f-65cd-6943-4ae998f756d7

PYTHONPATH=/path/to/grad-cam/vendor \
  /path/to/deploy_server/bundle_s2/env/bin/python \
  scripts/generate_stage2_gradcam_v8.py --data "$VISIONBOARD_DATA_ROOT"

/path/to/deploy_server/bundle/env/bin/python \
  scripts/generate_stage1_gradcam_v8.py --data "$VISIONBOARD_DATA_ROOT"
```

Stage2 attributes each retained ensemble box to its contributing YOLO8m/YOLO11l TTA views
and uses only the single contributor with the highest raw confidence; maps from the two
models are never merged. Stage1 uses the final fused head, explaining its combined
DINOv3 and convolutional-detail decision. Every retained TP and FP can therefore have a CAM;
FNs cannot, because no prediction exists to explain.

The standalone `grad-cam` package is needed only by the Stage2 generator. The GUI runtime
dependencies remain exactly those in `requirements.txt`.

## File formats

### Deployed Stage 1 and Stage 2

One normalized oriented box per line, exactly 10 fields:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 confidence
```

Each `x` is normalized by displayed-image width and each `y` by height. Class IDs use
`classes/stage1.txt` and `classes/stage2.txt`. Stage 1 normally maps to `object`. The eight
trained Stage 2 classes are `arty`, `camo`, `logistic`, `missile`, `radar`,
`smallvehicle`, `tank`, and `vehicle`.

### Raw Stage 2

Raw pre-calibration detections use absolute displayed-image coordinates:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name confidence
```

The GUI applies shipped thresholds and priority-aware cross-class deduplication at runtime,
so boxes, counts, arrows, and metrics use the same prediction set.

### Stage 2 vectors

Optional vector records contain 18 fields:

```text
x1 y1 x2 y2 x3 y3 x4 y4 winning_class winning_score s0 s1 s2 s3 s4 s5 s6 s7
```

Score order: `arty camo logistic missile radar smallvehicle tank vehicle`. Vectors affect
only the top-three information panel, never evaluation.

### `labels_simple`

The only ground truth used for evaluation:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

Coordinates are absolute displayed-image pixels. Axis-aligned boxes and rotated quads work.

### Delete regions

```text
x1 y1 x2 y2 ... xN yN Delete
```

These are ignore areas, not objects. They are always blacked out. Predictions whose centroid
falls inside are hidden and excluded from metrics and arrow navigation.

### Detailed labels and IDs

`labels_detail_id` has the normal 25-column detailed row plus the ID in column 26:

```text
x1 y1 x2 y2 x3 y3 x4 y4 super_category fine_label
Normal Snow Dark Quality Functionality Position P_Camo P_Cut Remarks User Is_Empty(ML)
SOC Camo_Instance Background_Matching difficult instance_id
```

Detailed rows are geometrically joined to `labels_simple`; they never change the evaluation
class. Direct box clicks show a compact summary. Arrow navigation exposes a collapsed
`labels_detailed` section that expands on demand.

The five-character ID is `IIcNN`: base-36 image number, lowercase class letter, and base-36
instance number within that image/class.

## Threshold and matching rules

Shipped Stage 2 thresholds are enabled by default:

| class | threshold |
|---|---:|
| radar | 0.39 |
| missile | 0.85 |
| arty | 0.86 |
| logistic | 0.77 |
| camo | 0.42 |
| smallvehicle | 0.24 |
| tank | 0.54 |
| vehicle | 0.84 |

`radar`, `missile`, `arty`, and `logistic` receive dedup priority. Cross-class boxes with
polygon IoU ≥ 0.40 collapse after thresholding. If a scene lacks raw Stage 2 scores,
per-class mode is unavailable for that scene rather than treating processed scores as raw.

Evaluation defaults to IoU 0.30:

- Stage 1 detection is class agnostic.
- Stage 2 detection is class agnostic.
- Stage 2 classification is class aware.
- Metrics stay visible regardless of visual-layer toggles.
- Prediction confidence handling is identical for boxes, counts, arrows, and metrics.

## Visualization behavior

- Stage 1 is blue and classed as `object`.
- Stage 2 retains all eight trained class names.
- GT boxes are dotted and class-colored.
- In evaluation mode, TP predictions are green and FP/wrong-class predictions are red.
  An FN has no prediction, so only its class-colored GT remains.
- With both stages visible, optional class-agnostic suppression hides Stage 1 boxes that
  overlap Stage 2 above IoU 0.30; residual Stage 1 boxes stay blue.
- Suppression affects visualization, not the separate Stage 1/2 metric panels.
- Class rows stay fixed even at zero count, ordered:
  `radar, missile, tank, arty, camo, logistic, smallvehicle, vehicle, object`.
- Class arrows traverse TP predictions, FN ground truth, and FP predictions for the active
  evaluation layer. A matched TP is focused through its prediction rather than its GT, so
  prediction details and Grad-CAM are immediately available.
- Direct clicks show a compact translucent panel. Arrow navigation provides detailed GT
  metadata or Stage 2 top-three class scores.

## Source layout

```text
app/                    FastAPI backend, parsing, evaluation, tiling, adapters
static/                 HTML/CSS/JS plus vendored OpenSeadragon
docs/                   data, architecture, and adapter documentation
requirements.txt        Python dependencies
run_gui.sh              launcher
```

Runtime content is created below `data/` and ignored by Git: tile pyramids, image-size cache,
jobs/logs, uploads, generated predictions, and runtime source registrations.

## API overview

| endpoint | purpose |
|---|---|
| `GET /api/scenes` | list scenes and available layers |
| `GET /api/instances` | search image/full instance IDs |
| `GET /api/scenes/{source}/{stem}/annotations` | boxes, GT, details, IDs, delete regions |
| `GET /api/scenes/{source}/{stem}/eval` | stage-specific evaluation |
| `GET /api/scenes/{source}/{stem}/tile_status` | build/poll a tile pyramid |
| `GET /api/scenes/{source}/{stem}/image.dzi` | Deep Zoom descriptor |
| `POST /api/infer` | enqueue adapter inference |
| `GET /api/jobs` | inspect jobs and logs |

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md),
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and
[docs/ADAPTERS.md](docs/ADAPTERS.md).

## Optional inference adapter

The repository includes the adapter code for the external YOLO+DINO inference engine, but
does not include that engine, its environment, or model weights. The V8 release starts in
viewer-only mode (`ENABLE_UPLOAD = False` and `ENABLE_INFERENCE = False` in
`app/config.py`). If inference is enabled, configure the external engine without editing
repository paths:

```bash
export VISIONBOARD_DINO_ROOT=/opt/dino_inference_model
export VISIONBOARD_DINO_PYTHON=/opt/dino-infer/bin/python
```

See `docs/ADAPTERS.md` for the adapter contract and output requirements.

## Development checks

```bash
python -m py_compile app/*.py app/adapters/*.py
bash run_gui.sh 7860
curl http://127.0.0.1:7860/api/scenes
```

The frontend has no build step. Edit files under `static/` and hard-refresh the browser.
