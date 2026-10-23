"""
Scene registry: discovers images / predictions / ground truth across configured sources.

A *scene* is one big image plus (optionally) a prediction file and a GT file. Scenes are
addressed as (source_name, stem). Nothing is copied — the store just resolves paths,
which keeps 50-megapixel datasets in place.

Image dimensions are read lazily from file headers (PIL doesn't decode pixels for .size)
and cached in data/cache/scene_meta.json keyed by path+mtime.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from PIL import Image

from . import config

Image.MAX_IMAGE_PIXELS = None  # our scenes are legitimately huge; disable the bomb guard

_meta_lock = threading.Lock()
_meta_cache: dict | None = None


def _load_meta_cache() -> dict:
    global _meta_cache
    if _meta_cache is None:
        try:
            _meta_cache = json.loads(config.META_CACHE.read_text())
        except Exception:
            _meta_cache = {}
    return _meta_cache


def _save_meta_cache() -> None:
    config.META_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.META_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_meta_cache))
    tmp.replace(config.META_CACHE)


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the image header, cached across restarts."""
    key = str(path)
    mtime = path.stat().st_mtime
    with _meta_lock:
        cache = _load_meta_cache()
        hit = cache.get(key)
        if hit and hit["mtime"] == mtime:
            return hit["w"], hit["h"]
    with Image.open(path) as im:
        w, h = im.size
    with _meta_lock:
        cache[key] = {"w": w, "h": h, "mtime": mtime}
        _save_meta_cache()
    return w, h


_sources_lock = threading.Lock()


def _load_user_sources() -> list[dict]:
    try:
        return json.loads(config.USER_SOURCES.read_text())
    except Exception:
        return []


def all_sources() -> list[dict]:
    """Built-in sources (app/config.py) + user-registered ones (data/sources.json)."""
    srcs = [dict(s, builtin=True) for s in config.SOURCES]
    names = {s["name"] for s in srcs}
    srcs += [dict(s, builtin=False) for s in _load_user_sources()
             if s.get("name") not in names]
    return srcs


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def add_user_source(spec: dict) -> dict:
    """Validate + persist a dataset registered from the GUI. Raises ValueError."""
    name = (spec.get("name") or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError("name must be letters/digits/_/- (no spaces)")
    images = Path((spec.get("images") or "").strip())
    if not images.is_dir():
        raise ValueError(f"images directory does not exist: {images}")

    def _dir(key):  # optional dirs — may be created later; empty means "none"
        v = (spec.get(key) or "").strip()
        return v or ""

    def _scale(key):
        try:
            v = float(spec.get(key, 1.0) or 1.0)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not 0 < v < 100:
            raise ValueError(f"{key} out of range")
        return v

    src = dict(
        name=name,
        title=(spec.get("title") or "").strip() or name,
        images=str(images),
        image_glob=(spec.get("image_glob") or "").strip() or "*",
        strip_suffix=(spec.get("strip_suffix") or "").strip(),
        preds=_dir("preds"),
        gt=_dir("gt"),
        delete=_dir("delete"),
        pred_scale=_scale("pred_scale"),
        gt_scale=_scale("gt_scale"),
        writable=False,
    )
    with _sources_lock:
        if any(s["name"] == name for s in all_sources()):
            raise ValueError(f"a source named '{name}' already exists")
        user = _load_user_sources()
        user.append(src)
        config.USER_SOURCES.parent.mkdir(parents=True, exist_ok=True)
        config.USER_SOURCES.write_text(json.dumps(user, indent=1))
    return src


def remove_user_source(name: str) -> bool:
    """Unregister a GUI-added dataset (files on disk are untouched)."""
    with _sources_lock:
        user = _load_user_sources()
        keep = [s for s in user if s.get("name") != name]
        if len(keep) == len(user):
            return False
        config.USER_SOURCES.write_text(json.dumps(keep, indent=1))
    return True


def get_source(name: str) -> dict | None:
    for s in all_sources():
        if s["name"] == name:
            return s
    return None


def _stem_of(src: dict, image_path: Path) -> str:
    stem = image_path.stem
    suf = src.get("strip_suffix") or ""
    if suf and stem.endswith(suf):
        stem = stem[: -len(suf)]
    return stem


def image_path(src: dict, stem: str) -> Path | None:
    """Find the image file for a scene stem (inverse of _stem_of)."""
    images = Path(src["images"])
    if not images.is_dir():
        return None
    suf = src.get("strip_suffix") or ""
    for p in sorted(images.glob(src.get("image_glob", "*"))):
        if p.is_file() and _stem_of(src, p) == stem:
            return p
    return None


def pred_path(src: dict, stem: str) -> Path | None:
    """Prediction file: source preds dir first, then the GUI's generated_preds fallback."""
    dirs = [Path(src["preds"])] if src.get("preds") else []
    dirs.append(config.GENERATED_PREDS / src["name"])
    for d in dirs:
        p = d / f"{stem}.txt"
        if p.is_file():
            return p
    return None


def _layer_path(src: dict, key: str, stem: str) -> Path | None:
    directory = src.get(key)
    if not directory:
        return None
    path = Path(directory) / f"{stem}.txt"
    return path if path.is_file() else None


def stage1_path(src: dict, stem: str) -> Path | None:
    return _layer_path(src, "stage1", stem)


def stage2_path(src: dict, stem: str) -> Path | None:
    # Legacy datasets expose their single prediction layer as Stage 2.
    return _layer_path(src, "stage2", stem) or pred_path(src, stem)


def stage2_raw_path(src: dict, stem: str) -> Path | None:
    return _layer_path(src, "stage2_raw", stem)


def stage2_vectors_path(src: dict, stem: str) -> Path | None:
    return _layer_path(src, "stage2_vectors", stem)


def stage0_path(src: dict, stem: str) -> Path | None:
    """Native-grid Stage0 heatmap stored as float16/float32 NumPy data."""
    directory = src.get("stage0")
    if not directory:
        return None
    path = Path(directory) / f"{stem}.npy"
    return path if path.is_file() else None


def gradcam_manifest_path(src: dict, stem: str) -> Path | None:
    """Per-scene manifest mapping stable prediction IDs to CAM images + bounds."""
    directory = src.get("gradcam")
    if not directory:
        return None
    path = Path(directory) / "manifests" / f"{stem}.json"
    return path if path.is_file() else None


def generated_pred_target(src: dict, stem: str) -> Path:
    """Where GUI-run inference should save the prediction for this scene."""
    if src.get("writable"):
        d = Path(src["preds"])
    else:
        d = config.GENERATED_PREDS / src["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.txt"


def gt_path(src: dict, stem: str) -> Path | None:
    d = src.get("gt")
    if not d:
        return None
    p = Path(d) / f"{stem}.txt"
    return p if p.is_file() else None


def delete_path(src: dict, stem: str) -> Path | None:
    """Delete-region polygon file (areas excluded from prediction display + eval)."""
    d = src.get("delete")
    if not d:
        return None
    p = Path(d) / f"{stem}.txt"
    return p if p.is_file() else None


def instance_id_path(src: dict, stem: str) -> Path | None:
    """Detailed annotation file whose final column is the stable instance ID."""
    d = src.get("instance_ids")
    if not d:
        return None
    p = Path(d) / f"{stem}.txt"
    return p if p.is_file() else None


def list_scenes() -> list[dict]:
    """Every scene across all sources, with dims + pred/GT availability."""
    out = []
    for src in all_sources():
        images = Path(src["images"])
        if not images.is_dir():
            continue
        for p in sorted(images.glob(src.get("image_glob", "*"))):
            if not p.is_file() or p.suffix.lower() not in config.ALLOWED_UPLOAD_EXT:
                continue
            stem = _stem_of(src, p)
            try:
                w, h = image_size(p)
            except Exception:
                continue  # unreadable/corrupt image — skip rather than break the listing
            out.append(dict(
                source=src["name"],
                source_title=src["title"],
                stem=stem,
                image=p.name,
                width=w,
                height=h,
                has_stage1=stage1_path(src, stem) is not None,
                has_stage2=stage2_path(src, stem) is not None,
                has_stage2_raw=stage2_raw_path(src, stem) is not None,
                has_stage2_vectors=stage2_vectors_path(src, stem) is not None,
                has_stage0=stage0_path(src, stem) is not None,
                has_gradcam=gradcam_manifest_path(src, stem) is not None,
                has_pred=(stage1_path(src, stem) is not None or
                          stage2_path(src, stem) is not None),
                has_gt=gt_path(src, stem) is not None,
            ))
    return out


def get_scene(source: str, stem: str) -> dict | None:
    """Full record for one scene, or None if it doesn't exist."""
    src = get_source(source)
    if not src:
        return None
    img = image_path(src, stem)
    if not img:
        return None
    w, h = image_size(img)
    return dict(
        source=source,
        stem=stem,
        image_path=img,
        width=w,
        height=h,
        pred_path=pred_path(src, stem),
        stage1_path=stage1_path(src, stem),
        stage2_path=stage2_path(src, stem),
        stage2_raw_path=stage2_raw_path(src, stem),
        stage2_vectors_path=stage2_vectors_path(src, stem),
        stage0_path=stage0_path(src, stem),
        gradcam_manifest_path=gradcam_manifest_path(src, stem),
        gradcam_root=(Path(src["gradcam"]) if src.get("gradcam") else None),
        review_root=(Path(src["reviews"]) if src.get("reviews") else
                     Path(src["images"]).parent / "reviews"),
        gt_path=gt_path(src, stem),
        delete_path=delete_path(src, stem),
        instance_id_path=instance_id_path(src, stem),
        pred_scale=float(src.get("pred_scale", 1.0)),
        gt_scale=float(src.get("gt_scale", 1.0)),
        instance_id_scale=float(src.get("instance_id_scale", src.get("gt_scale", 1.0))),
        prediction_format=src.get("prediction_format", "dota"),
        stage1_classes=src.get("stage1_classes"),
        stage2_classes=src.get("stage2_classes"),
        stage2_raw_format=src.get("stage2_raw_format", "dota"),
        stage2_raw_scale=float(src.get("stage2_raw_scale", 1.0)),
        stage2_is_shipped=bool(src.get("stage2_is_shipped", False)),
        stage2_thresholds=dict(src.get("stage2_thresholds") or {}),
        stage2_priority=list(src.get("stage2_priority") or []),
        stage2_dedup_iou=float(src.get("stage2_dedup_iou", 0.40)),
        stage_suppress_iou=float(src.get("stage_suppress_iou", 0.30)),
    )
