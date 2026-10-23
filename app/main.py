"""
FastAPI app — HTTP surface of the GUI. Run with:  bash run_gui.sh   (see README.md)

Route map (all JSON unless noted):
  GET  /                                          the single-page UI (static/index.html)
  GET  /api/sources                               list datasets (built-in + user-registered)
  POST /api/sources                               register a dataset (image/pred/GT folders)
  DEL  /api/sources/{name}                        unregister a user-added dataset
  GET  /api/scenes                                list all scenes across sources
  GET  /api/scenes/{source}/{stem}                one scene's metadata
  GET  /api/scenes/{source}/{stem}/annotations    preds + GT, coords in image pixel space
  GET  /api/scenes/{source}/{stem}/eval           TP/FP/FN matching (?iou=&conf=&ignore_difficult=)
  GET  /api/scenes/{source}/{stem}/tile_status    pyramid state; kicks off a build if needed
  GET  /api/scenes/{source}/{stem}/image.dzi      DZI descriptor (XML) once the pyramid is ready
  GET  /api/scenes/{source}/{stem}/image_files/{level}/{col}_{row}.{ext}   one pyramid tile
  POST /api/upload                                multipart image upload -> 'uploads' source
  GET  /api/adapters                              available model adapters
  POST /api/infer                                 queue an inference job {source, stem, adapter, device, extra}
  GET  /api/jobs                                  recent jobs
  GET  /api/jobs/{job_id}                         job state + log tail
"""
from __future__ import annotations

import io
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import annotations as ann
from . import config, confusion, evaluate, jobs, reviews as review_store, store, tiler
from .adapters import ADAPTERS

app = FastAPI(title="VisionBoard", docs_url="/api/docs")


@lru_cache(maxsize=8)
def _confusion_report(source: str, pcamo_mode: str) -> dict:
    def loader(scene: dict):
        _stage1, stage2, regions = _prepared_predictions(
            scene, config.DEFAULT_CONF_THR, "per_class", True, include_vectors=False)
        return stage2, regions
    return confusion.build(source, pcamo_mode, loader, config.DEFAULT_IOU_THR)


def _gradcam_manifest(scene: dict) -> dict:
    path = scene.get("gradcam_manifest_path")
    if not path:
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return data.get("predictions", data) if isinstance(data, dict) else {}


def _attach_gradcam(predictions: list[dict], scene: dict) -> None:
    manifest = _gradcam_manifest(scene)
    for prediction in predictions:
        entry = manifest.get(prediction["id"])
        if isinstance(entry, dict):
            prediction["gradcam"] = {
                key: entry[key] for key in (
                    "bounds", "models", "contributors", "method", "status"
                ) if key in entry
            }
            prediction["gradcam"]["url"] = (
                f"/api/scenes/{scene['source']}/{scene['stem']}/gradcam"
                f"?prediction_id={prediction['id']}"
            )


def _scene_or_404(source: str, stem: str) -> dict:
    scene = store.get_scene(source, stem)
    if not scene:
        raise HTTPException(404, f"scene {source}/{stem} not found")
    return scene


def _review_instance_or_404(source: str, stem: str, instance_id: str) -> dict:
    """Return the requested GT and prevent reviews from attaching to predictions/unknown IDs."""
    scene = _scene_or_404(source, stem)
    gt = ann.load_gt(scene["gt_path"], scene["gt_scale"]) if scene["gt_path"] else []
    ann.attach_instance_ids(gt, scene["instance_id_path"], scene["instance_id_scale"])
    match = next((item for item in gt if item.get("instance_id") == instance_id), None)
    if not match:
        raise HTTPException(404, f"GT instance {instance_id} not found in {source}/{stem}")
    return match


def _class_names(path_value) -> list[str] | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path.read_text().splitlines() if path.is_file() else None


def _prediction_layers(scene: dict, conf: float = config.DEFAULT_CONF_THR,
                       confidence_mode: str = "global",
                       include_vectors: bool = True) -> tuple[list[dict], list[dict]]:
    common = dict(fmt=scene["prediction_format"], width=scene["width"], height=scene["height"])
    stage1 = (ann.load_preds(scene["stage1_path"], scene["pred_scale"],
                             class_names=_class_names(scene["stage1_classes"]),
                             stage="stage1", **common)
              if scene["stage1_path"] else [])
    stage1 = [p for p in stage1 if p["score"] >= conf]

    if scene["stage2_raw_path"]:
        stage2 = ann.load_preds(
            scene["stage2_raw_path"], scene["stage2_raw_scale"],
            fmt=scene["stage2_raw_format"], width=scene["width"], height=scene["height"],
            class_names=_class_names(scene["stage2_classes"]), stage="stage2")
        thresholds = (scene["stage2_thresholds"] if confidence_mode == "per_class"
                      else {p["cls"]: conf for p in stage2})
        stage2 = ann.threshold_and_dedup(
            stage2, thresholds, scene["stage2_priority"], scene["stage2_dedup_iou"])
    else:
        if confidence_mode == "per_class" and not scene.get("stage2_is_shipped"):
            raise ValueError("raw Stage-2 confidence scores are unavailable for this scene")
        stage2 = (ann.load_preds(scene["stage2_path"], scene["pred_scale"],
                                 class_names=_class_names(scene["stage2_classes"]),
                                 stage="stage2", **common)
                  if scene["stage2_path"] else [])
        # Deployed files have already passed the shipped per-class operating point.
        # Do not apply a second threshold to their calibrated display scores.
        if confidence_mode != "per_class":
            stage2 = [p for p in stage2 if p["score"] >= conf]
    if include_vectors:
        vectors = ann.load_prediction_vectors(scene.get("stage2_vectors_path"), 1.0)
        ann.attach_top_classes(stage2, vectors)
    return stage1, stage2


def _prepared_predictions(scene: dict, conf: float, confidence_mode: str,
                          suppress_stage1: bool,
                          include_vectors: bool = True) -> tuple[list[dict], list[dict], list[dict]]:
    try:
        stage1, stage2 = _prediction_layers(scene, conf, confidence_mode, include_vectors)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    regions = (ann.load_delete_regions(scene["delete_path"], scene["gt_scale"])
               if scene["delete_path"] else [])
    ann.mark_in_delete(stage1, regions)
    ann.mark_in_delete(stage2, regions)
    if suppress_stage1:
        ann.suppress_stage1(stage1, stage2, scene["stage_suppress_iou"])
    return stage1, stage2, regions


# ── sources (datasets) ──────────────────────────────────────────────────────────────────
@app.get("/api/sources")
def api_sources():
    """All datasets; user-registered ones (builtin=False) can be removed via DELETE."""
    return dict(sources=[
        {k: s.get(k) for k in ("name", "title", "images", "image_glob", "strip_suffix",
                               "preds", "stage1", "stage2", "prediction_format",
                               "stage1_classes", "stage2_classes", "stage_suppress_iou",
                               "gt", "delete", "instance_ids", "pred_scale", "gt_scale",
                               "instance_id_scale", "builtin")}
        for s in store.all_sources()
    ])


@app.post("/api/sources")
def api_add_source(body: dict):
    """Register a dataset at runtime: point at image/pred/GT folders you prepared."""
    try:
        src = store.add_user_source(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return src


@app.delete("/api/sources/{name}")
def api_remove_source(name: str):
    if not store.remove_user_source(name):
        raise HTTPException(404, f"no user-registered source named '{name}' "
                                 "(built-in sources are removed in app/config.py)")
    return dict(removed=name)


# ── scenes ───────────────────────────────────────────────────────────────────────────────
@app.get("/api/scenes")
def api_scenes():
    return dict(scenes=store.list_scenes(),
                ui=dict(upload=config.ENABLE_UPLOAD, inference=config.ENABLE_INFERENCE))


@app.get("/api/confusion")
def api_confusion(source: str = Query("test85_oct2026"),
                  pcamo_mode: str = Query("simple", pattern="^(simple|actual)$")):
    if not store.get_source(source):
        raise HTTPException(404, f"source {source} not found")
    return _confusion_report(source, pcamo_mode)


@app.get("/api/instances")
def api_instances(q: str = Query(..., min_length=1),
                  mode: str = Query("full", pattern="^(full|image|instance)$"),
                  limit: int = Query(1000, ge=1, le=2000)):
    """Find stable GT instance IDs across all configured scenes."""
    query = q.strip().lower()
    if not query:
        return dict(results=[])
    results = []
    image_results = {}
    for scene in store.list_scenes():
        record = store.get_scene(scene["source"], scene["stem"])
        path = record.get("instance_id_path") if record else None
        if not path:
            continue
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 10:
                continue
            instance_id = parts[-1]
            iid = instance_id.lower()
            if mode == "full":
                matched = query in iid
            elif mode == "image":
                matched = query == iid[:2]
            else:
                matched = query == iid[-2:]
            if not matched:
                continue
            result = dict(source=scene["source"], stem=scene["stem"],
                          instance_id=instance_id, cls=parts[8],
                          image_number=instance_id[:2], instance_number=instance_id[-2:])
            if mode == "image":
                image_results[(scene["source"], scene["stem"])] = dict(
                    source=scene["source"], stem=scene["stem"],
                    instance_id=None, cls=None, image_number=instance_id[:2])
            else:
                results.append(result)
    if mode == "image":
        results = list(image_results.values())
    results.sort(key=lambda r: (r["instance_id"].lower() != query,
                                r["instance_id"].lower(), r["stem"])
                 if r["instance_id"] else (False, r["image_number"].lower(), r["stem"]))
    return dict(results=results[:limit], total=len(results))


@app.get("/api/scenes/{source}/{stem}")
def api_scene(source: str, stem: str):
    s = _scene_or_404(source, stem)
    return dict(source=s["source"], stem=s["stem"], width=s["width"], height=s["height"],
                image=s["image_path"].name,
                has_stage1=s["stage1_path"] is not None,
                has_stage2=s["stage2_path"] is not None,
                has_stage2_raw=s["stage2_raw_path"] is not None,
                stage2_is_shipped=s.get("stage2_is_shipped", False),
                has_stage2_vectors=s["stage2_vectors_path"] is not None,
                has_stage0=s["stage0_path"] is not None,
                has_gradcam=s["gradcam_manifest_path"] is not None,
                has_pred=(s["stage1_path"] is not None or s["stage2_path"] is not None),
                has_gt=s["gt_path"] is not None,
                stage_suppress_iou=s["stage_suppress_iou"])


@app.get("/api/scenes/{source}/{stem}/annotations")
def api_annotations(source: str, stem: str,
                    conf: float = Query(config.DEFAULT_CONF_THR, ge=0.0),
                    confidence_mode: str = Query("global", pattern="^(global|per_class)$"),
                    suppress_stage1: bool = True):
    s = _scene_or_404(source, stem)
    stage1, stage2, regions = _prepared_predictions(
        s, conf, confidence_mode, suppress_stage1)
    gt = ann.load_gt(s["gt_path"], s["gt_scale"]) if s["gt_path"] else []
    ann.attach_instance_ids(gt, s["instance_id_path"], s["instance_id_scale"])
    ann.mark_in_delete(gt, regions)
    gt = [item for item in gt if not item.get("in_delete")]
    _attach_gradcam(stage1, s)
    _attach_gradcam(stage2, s)
    combined = [p for p in stage1 if not p.get("suppressed_by_stage2")] + stage2
    return dict(stage1=stage1, stage2=stage2, preds=combined, gt=gt,
                delete_regions=regions,
                suppressed_stage1=sum(bool(p.get("suppressed_by_stage2")) for p in stage1),
                stage_suppress_iou=s["stage_suppress_iou"])


@app.get("/api/scenes/{source}/{stem}/eval")
def api_eval(source: str, stem: str,
             iou: float = Query(config.DEFAULT_IOU_THR, ge=0.05, le=0.95),
             conf: float = Query(config.DEFAULT_CONF_THR, ge=0.0),
             ignore_difficult: bool = True,
             prediction_set: str = Query("combined"),
             class_agnostic: bool = False,
             confidence_mode: str = Query("global", pattern="^(global|per_class)$"),
             suppress_stage1: bool = True):
    s = _scene_or_404(source, stem)
    if not s["gt_path"]:
        raise HTTPException(400, "scene needs both predictions and ground truth for evaluation")
    stage1, stage2, regions = _prepared_predictions(
        s, conf, confidence_mode, suppress_stage1, include_vectors=False)
    if prediction_set == "stage1":
        preds = stage1
    elif prediction_set == "stage2":
        preds = stage2
    elif prediction_set == "combined":
        preds = [p for p in stage1 if not p.get("suppressed_by_stage2")] + stage2
    else:
        raise HTTPException(400, "prediction_set must be stage1, stage2, or combined")
    if not preds and not (s["stage1_path"] or s["stage2_path"]):
        raise HTTPException(400, "scene has no prediction file")
    gt = ann.load_gt(s["gt_path"], s["gt_scale"])
    ann.mark_in_delete(gt, regions)
    gt = [item for item in gt if not item.get("in_delete")]
    preds = [p for p in preds if not p.get("in_delete")]
    return evaluate.evaluate(preds, gt, iou_thr=iou, conf_thr=0.0,
                             ignore_difficult=ignore_difficult,
                             class_agnostic=class_agnostic)


# ── timestamped GT reviews ───────────────────────────────────────────────────────────────
@app.get("/api/scenes/{source}/{stem}/reviews")
def api_reviews(source: str, stem: str, instance_id: str = Query(..., min_length=1)):
    _review_instance_or_404(source, stem, instance_id)
    try:
        return review_store.list_instance(source, stem, instance_id)
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/scenes/{source}/{stem}/reviews")
def api_create_review(source: str, stem: str, body: dict):
    instance_id = str(body.get("instance_id") or "").strip()
    text = str(body.get("text") or "")
    if not instance_id:
        raise HTTPException(400, "instance_id is required")
    _review_instance_or_404(source, stem, instance_id)
    try:
        return review_store.create(source, stem, instance_id, text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/scenes/{source}/{stem}/reviews/{entry_id}")
def api_update_review(source: str, stem: str, entry_id: str, body: dict):
    instance_id = str(body.get("instance_id") or "").strip()
    text = str(body.get("text") or "")
    if not instance_id:
        raise HTTPException(400, "instance_id is required")
    _review_instance_or_404(source, stem, instance_id)
    try:
        return review_store.update(source, stem, instance_id, entry_id, text)
    except KeyError as exc:
        raise HTTPException(404, f"review entry {entry_id} not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/scenes/{source}/{stem}/reviews/{entry_id}")
def api_delete_review(source: str, stem: str, entry_id: str,
                      instance_id: str = Query(..., min_length=1)):
    _review_instance_or_404(source, stem, instance_id)
    try:
        return review_store.remove(source, stem, instance_id, entry_id)
    except KeyError as exc:
        raise HTTPException(404, f"review entry {entry_id} not found") from exc


@app.get("/api/reviews/export")
def api_export_reviews(source: str = Query("test85_oct2026"),
                       format: str = Query("json", pattern="^(json|csv)$")):
    if not store.get_source(source):
        raise HTTPException(404, f"source {source} not found")
    try:
        body, media_type, filename = review_store.export(source, format)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(content=body, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/scenes/{source}/{stem}/stage0.png")
def api_stage0(source: str, stem: str):
    """Colorized full-scene Stage0 heatmap; browser scales native grid to display grid."""
    import cv2
    import numpy as np

    scene = _scene_or_404(source, stem)
    path = scene.get("stage0_path")
    if not path:
        raise HTTPException(404, "Stage0 heatmap unavailable")
    heat = np.load(path).astype(np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)
    heat = np.clip(heat, 0.0, 1.0)
    color = cv2.applyColorMap(np.uint8(heat * 255.0), cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".png", color)
    if not ok:
        raise HTTPException(500, "could not encode Stage0 heatmap")
    return Response(content=encoded.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/scenes/{source}/{stem}/gradcam")
def api_gradcam(source: str, stem: str, prediction_id: str = Query(...)):
    """Selected prediction's precomputed, spatially aligned CAM image."""
    scene = _scene_or_404(source, stem)
    entry = _gradcam_manifest(scene).get(prediction_id)
    root = scene.get("gradcam_root")
    if not isinstance(entry, dict) or not entry.get("file") or root is None:
        raise HTTPException(404, "Grad-CAM unavailable for this prediction")
    root = root.resolve()
    path = (root / entry["file"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, "invalid Grad-CAM path") from exc
    if not path.is_file():
        raise HTTPException(404, "Grad-CAM image is missing")
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "no-store"})


# ── tiles ────────────────────────────────────────────────────────────────────────────────
@app.get("/api/scenes/{source}/{stem}/tile_status")
def api_tile_status(source: str, stem: str):
    s = _scene_or_404(source, stem)
    return tiler.ensure_build(source, stem, s["image_path"],
                              s.get("delete_path"), s.get("gt_scale", 1.0))


@app.get("/api/scenes/{source}/{stem}/image.dzi")
def api_dzi(source: str, stem: str):
    _scene_or_404(source, stem)
    xml = tiler.dzi_descriptor(source, stem)
    if xml is None:
        raise HTTPException(409, "tile pyramid not built yet — poll tile_status")
    return Response(content=xml, media_type="application/xml")


# NOTE: path must be <dzi basename>_files/... — OpenSeadragon derives tile URLs
# from the .dzi URL per the Deep Zoom convention (image.dzi -> image_files/).
@app.get("/api/scenes/{source}/{stem}/image_files/{level}/{name}")
def api_tile(source: str, stem: str, level: int, name: str):
    m = re.fullmatch(r"(\d+)_(\d+)\.(\w+)", name)
    if not m:
        raise HTTPException(400, "bad tile name")
    p = tiler.tile_path(source, stem, level, int(m.group(1)), int(m.group(2)))
    if not p.is_file():
        raise HTTPException(404, "tile not found")
    return FileResponse(p, headers={"Cache-Control": "public, max-age=86400"})


# ── upload ───────────────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.png").name  # strip any client path
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_UPLOAD_EXT:
        raise HTTPException(400, f"unsupported extension {ext}; allowed: "
                                 f"{sorted(config.ALLOWED_UPLOAD_EXT)}")
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOADS_DIR / name
    if dest.exists():
        raise HTTPException(409, f"an upload named '{name}' already exists")
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > config.MAX_UPLOAD_MB << 20:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file exceeds {config.MAX_UPLOAD_MB} MB limit")
            f.write(chunk)
    try:
        w, h = store.image_size(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "file is not a readable image")
    return dict(source="uploads", stem=dest.stem, width=w, height=h)


# ── inference ────────────────────────────────────────────────────────────────────────────
@app.get("/api/adapters")
def api_adapters():
    return dict(adapters=[a.to_json() for a in ADAPTERS.values()])


@app.post("/api/infer")
def api_infer(body: dict):
    for k in ("source", "stem", "adapter"):
        if not body.get(k):
            raise HTTPException(400, f"missing field '{k}'")
    try:
        job = jobs.submit(body["source"], body["stem"], body["adapter"],
                          device=str(body.get("device", "0")),
                          extra=body.get("extra") or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return job


@app.get("/api/jobs")
def api_jobs():
    return dict(jobs=jobs.list_jobs())


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    job["log_tail"] = jobs.log_tail(job_id)
    return job


# ── static UI (mounted last so /api and /tiles win) ─────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return HTMLResponse((config.STATIC / "index.html").read_text())


@app.get("/confusion", include_in_schema=False)
def confusion_page():
    return HTMLResponse((config.STATIC / "confusion.html").read_text())


app.mount("/static", StaticFiles(directory=str(config.STATIC)), name="static")


def main():
    import uvicorn
    for d in (config.UPLOADS_DIR, config.GENERATED_PREDS, config.TILE_CACHE,
              config.JOBS_DIR, config.META_CACHE.parent):
        d.mkdir(parents=True, exist_ok=True)
    if shutil.which("nvidia-smi"):
        pass  # informational only; device choice is per-job
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
