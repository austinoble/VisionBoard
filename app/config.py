"""
gui_pipeline configuration — the single place you edit to point the GUI at data and models.

The GUI is model-agnostic: it displays big images (as deep-zoom tile pyramids), overlays
DOTA-style prediction / ground-truth boxes as vectors, and evaluates TP/FP/FN. Everything
dataset- or model-specific lives here or in app/adapters/.

──────────────────────────────────────────────────────────────────────────────────────────
COORDINATE CONTRACT (read this before adding a source or adapter)
──────────────────────────────────────────────────────────────────────────────────────────
The GUI displays annotations in the *pixel space of the displayed image file*.
Stored annotation files may be in a different space; each source declares multipliers:

    displayed_xy = stored_xy * pred_scale   (for prediction files)
    displayed_xy = stored_xy * gt_scale     (for ground-truth files)

In the bundled V7 source definition, displayed images are the 2x super-resolved PNGs.
Deployed predictions and `labels_simple` have already been converted to that displayed
space, so pred_scale=1.0 and gt_scale=1.0. The detailed instance-ID labels remain in native
coordinates and use instance_id_scale=2.0.

──────────────────────────────────────────────────────────────────────────────────────────
SOURCES
──────────────────────────────────────────────────────────────────────────────────────────
Each source is one dataset the GUI can browse. Fields:

  name          unique id, used in URLs — letters/digits/_/- only
  title         human-readable label shown in the UI
  images        directory of scene images (png/jpg/tif)
  image_glob    glob pattern selecting scene images inside `images`
  strip_suffix  removed from the image stem to obtain the scene *stem* used to find the
                matching pred/GT file (e.g. "abc_SR2x.png" + strip_suffix="_SR2x" ->
                pred/GT file "abc.txt")
  preds         directory of prediction files, "<stem>.txt" DOTA format (may not exist yet)
  gt            directory of ground-truth files, "<stem>.txt" DOTA format (optional)
  delete        directory of delete-region files, "<stem>.txt" (optional): N-point polygons
                marking excluded areas ("x1 y1 ... xN yN Delete" per line). Scaled with
                gt_scale. Regions are blacked out; predictions whose centre falls inside
                are hidden and excluded from evaluation and navigation.
  instance_ids  detailed GT files whose final column is a stable instance ID (optional)
  pred_scale    stored-pred -> displayed-image coordinate multiplier
  gt_scale      stored-GT   -> displayed-image coordinate multiplier
  instance_id_scale  stored detailed-ID coords -> displayed-image coordinate multiplier
  writable      if True, "Run inference" saves predictions into `preds`; otherwise results
                go to data/generated_preds/<source>/ (which is always checked as a fallback
                for every source, so read-only datasets still get overlays)

Predictions produced from the GUI are ALWAYS also discoverable via the fallback directory,
so you never need write access to a dataset to use it.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC = ROOT / "static"

UPLOADS_DIR = DATA / "uploads" / "images"              # user-uploaded scenes land here
USER_SOURCES = DATA / "sources.json"                   # datasets registered from the GUI (merged with SOURCES below)
GENERATED_PREDS = DATA / "generated_preds"             # GUI-run inference output (per source)
TILE_CACHE = DATA / "tile_cache"                       # deep-zoom pyramids, safe to delete
JOBS_DIR = DATA / "jobs"                               # inference job logs + work dirs
META_CACHE = DATA / "cache" / "scene_meta.json"        # cached image dimensions

# ── GUI features ─────────────────────────────────────────────────────────────────────────
# Viewer-only mode: the current pred/GT sets are pre-computed, so uploading and running
# inference from the browser are switched off. Set back to True to re-enable the buttons.
ENABLE_UPLOAD = False
ENABLE_INFERENCE = False

# ── Datasets shown in the GUI ────────────────────────────────────────────────────────────
# Only entries in SOURCES appear in the sidebar. Parked entries live in _DISABLED_SOURCES
# below — move a dict between the two lists (and restart) to show/hide a dataset.
LOWRES_DEMO = Path(os.environ.get(
    "VISIONBOARD_DATA_ROOT",
    "/home/jovyan/SFS/CVW/AutoRA/DATA/LR_Data_06oct26/VisionBoard_Test85_Oct2026",
)).expanduser()
LOWRES_RAW_STAGE2 = Path(os.environ.get(
    "VISIONBOARD_STAGE2_RAW", str(LOWRES_DEMO / "predictions" / "stage2_raw"))).expanduser()
LOWRES_STAGE2_VECTORS = Path(os.environ.get(
    "VISIONBOARD_STAGE2_VECTORS",
    str(LOWRES_DEMO / "predictions" / "stage2_vectors"))).expanduser()
LOWRES_STAGE0 = Path(os.environ.get(
    "VISIONBOARD_STAGE0_ROOT", str(LOWRES_DEMO / "heatmaps" / "stage0"))).expanduser()
LOWRES_GRADCAM = Path(os.environ.get(
    "VISIONBOARD_GRADCAM_ROOT", str(LOWRES_DEMO / "explainability" / "gradcam"))).expanduser()
LOWRES_STAGE2_THRESHOLDS = {
    "radar": 0.39, "missile": 0.85, "arty": 0.86, "logistic": 0.77,
    "camo": 0.42, "smallvehicle": 0.24, "tank": 0.54, "vehicle": 0.84,
}
SOURCES = [
    dict(
        name=os.environ.get("VISIONBOARD_SOURCE_NAME", "test85_oct2026"),
        title=os.environ.get("VISIONBOARD_DATA_TITLE", "October 2026 — Test85"),
        images=str(LOWRES_DEMO / "images"),
        image_glob="*.png",
        strip_suffix="",
        stage1=str(LOWRES_DEMO / "predictions" / "stage1"),
        stage2=str(LOWRES_DEMO / "predictions" / "stage2"),
        stage2_is_shipped=True,
        stage2_raw=str(LOWRES_RAW_STAGE2),
        stage2_vectors=str(LOWRES_STAGE2_VECTORS),
        stage2_raw_format="dota",
        stage2_raw_scale=1.0,
        stage2_thresholds=LOWRES_STAGE2_THRESHOLDS,
        stage2_priority=["radar", "missile", "arty", "logistic"],
        stage2_dedup_iou=0.40,
        prediction_format="lowres_normalized",
        stage1_classes=str(LOWRES_DEMO / "classes" / "stage1.txt"),
        stage2_classes=str(LOWRES_DEMO / "classes" / "stage2.txt"),
        stage_suppress_iou=0.30,
        gt=str(LOWRES_DEMO / "gt" / "labels_simple"),
        delete=str(LOWRES_DEMO / "gt" / "labels_delete"),
        instance_ids=str(LOWRES_DEMO / "gt" / "labels_detailed"),
        stage0=str(LOWRES_STAGE0),
        gradcam=str(LOWRES_GRADCAM),
        reviews=str(LOWRES_DEMO / "reviews"),
        pred_scale=1.0,
        gt_scale=1.0,
        instance_id_scale=1.0,
        writable=False,
    ),
]

_DISABLED_SOURCES = []

# ── Tiling ───────────────────────────────────────────────────────────────────────────────
TILE_SIZE = 512          # deep-zoom tile edge, px
TILE_OVERLAP = 1         # DZI overlap, px (helps hide seams)
TILE_FORMAT = "png"      # lossless tiles for clear zoomed inspection
TILE_JPEG_QUALITY = 92

# ── Score normalization ──────────────────────────────────────────────────────────────────
# Composite/fused scores can exceed 1.0 with class-dependent scales (e.g. the ensemble's
# missile sum-rule reaches ~5). For display + the confidence slider + evaluation, scores
# are mapped into 0..1:  per class, if that class's max score in the file is > 1.0, all its
# scores are divided by that max; classes already within 0..1 are left untouched.
# The raw score is still shown in the box-info popup. Set to None to disable entirely,
# or "global" to divide by the file-wide max instead.
NORMALIZE_SCORES = "per_class"      # "per_class" | "global" | None

# ── Evaluation defaults (client can override per request) ───────────────────────────────
DEFAULT_IOU_THR = 0.30
DEFAULT_CONF_THR = 0.30

# ── Upload limits ────────────────────────────────────────────────────────────────────────
ALLOWED_UPLOAD_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_MB = 4096

# ── Server ───────────────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 7860
