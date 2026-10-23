"""
Deep-zoom (DZI) tile pyramid builder + reader.

Why: scenes here are 15–60+ megapixels. Shipping the raw PNG to the browser would mean
hundreds of MB of decode per view. Instead we pre-cut each scene once into a standard
Deep Zoom pyramid (the format OpenSeadragon speaks natively): level N is the full image,
each level below is a 2x downscale, every level is cut into TILE_SIZE tiles. The viewer
then only ever fetches the handful of tiles covering the current viewport at the current
zoom — pan/zoom over a gigantic scene stays smooth and the image is shown at its TRUE
pixel dimensions (level max == original resolution, nothing resampled at 1:1).

Pyramids are cached in data/tile_cache/<source>__<stem>/ and invalidated when the source
image's mtime changes. Building a ~50 MP scene takes ~10–30 s and happens once, in a
background thread; the API exposes build progress so the UI can show it.

Layout on disk (per scene):
    meta.json                     {width, height, tile_size, overlap, format, src_mtime}
    files/<level>/<col>_<row>.jpg
"""
from __future__ import annotations

import json
import math
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np

from . import config

# in-flight build progress: key -> {"state": "building"|"error", "progress": float, "error": str}
_builds: dict[str, dict] = {}
_builds_lock = threading.Lock()


def cache_key(source: str, stem: str) -> str:
    return f"{source}__{stem}"


def pyramid_dir(source: str, stem: str) -> Path:
    return config.TILE_CACHE / cache_key(source, stem)


def _meta_path(pdir: Path) -> Path:
    return pdir / "meta.json"


def max_level(w: int, h: int) -> int:
    return max(0, math.ceil(math.log2(max(w, h))))


def status(source: str, stem: str, image_path: Path,
           delete_path: Path | None = None, delete_scale: float = 1.0) -> dict:
    """{'state': 'ready'|'building'|'none'|'error', 'progress': float}"""
    key = cache_key(source, stem)
    with _builds_lock:
        b = _builds.get(key)
        if b:
            return dict(b)
    pdir = pyramid_dir(source, stem)
    mp = _meta_path(pdir)
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text())
            delete_mtime = delete_path.stat().st_mtime if delete_path and delete_path.is_file() else None
            if (meta.get("src_mtime") == image_path.stat().st_mtime
                    and meta.get("delete_mtime") == delete_mtime
                    and float(meta.get("delete_scale", 1.0)) == float(delete_scale)):
                return {"state": "ready", "progress": 1.0}
        except Exception:
            pass
    return {"state": "none", "progress": 0.0}


def ensure_build(source: str, stem: str, image_path: Path,
                 delete_path: Path | None = None, delete_scale: float = 1.0) -> dict:
    """Kick off a pyramid build in the background if one isn't ready/in-flight."""
    st = status(source, stem, image_path, delete_path, delete_scale)
    if st["state"] in ("ready", "building"):
        return st
    key = cache_key(source, stem)
    with _builds_lock:
        if key in _builds:  # raced another request
            return dict(_builds[key])
        _builds[key] = {"state": "building", "progress": 0.0}
    t = threading.Thread(target=_build,
                         args=(source, stem, image_path, delete_path, delete_scale), daemon=True)
    t.start()
    return {"state": "building", "progress": 0.0}


def _set_progress(key: str, p: float) -> None:
    with _builds_lock:
        if key in _builds:
            _builds[key]["progress"] = round(p, 3)


def _build(source: str, stem: str, image_path: Path,
           delete_path: Path | None = None, delete_scale: float = 1.0) -> None:
    key = cache_key(source, stem)
    pdir = pyramid_dir(source, stem)
    try:
        shutil.rmtree(pdir, ignore_errors=True)
        (pdir / "files").mkdir(parents=True)

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # BGR; cv2 writes BGR -> correct jpg
        if img is None:
            raise RuntimeError(f"cv2 could not read {image_path}")
        # Bake ignore regions into display tiles. Source imagery is never modified, and
        # deleted content cannot flash through while the vector canvas catches up to zoom.
        if delete_path and delete_path.is_file():
            for line in delete_path.read_text().splitlines():
                values = []
                for token in line.split():
                    try:
                        values.append(float(token) * delete_scale)
                    except ValueError:
                        break
                if len(values) >= 6:
                    if len(values) % 2:
                        values = values[:-1]
                    points = np.asarray(list(zip(values[0::2], values[1::2])), dtype=np.int32)
                    cv2.fillPoly(img, [points], color=(0, 0, 0))
        h, w = img.shape[:2]
        top = max_level(w, h)
        ts, ov = config.TILE_SIZE, config.TILE_OVERLAP
        ext = config.TILE_FORMAT
        jpg_params = [cv2.IMWRITE_JPEG_QUALITY, config.TILE_JPEG_QUALITY] if ext == "jpg" else []

        # total tile count for progress reporting
        total = done = 0
        lw, lh = w, h
        for _ in range(top, -1, -1):
            total += math.ceil(lw / ts) * math.ceil(lh / ts)
            lw, lh = max(1, math.ceil(lw / 2)), max(1, math.ceil(lh / 2))

        level_img = img
        for level in range(top, -1, -1):
            lh_, lw_ = level_img.shape[:2]
            ldir = pdir / "files" / str(level)
            ldir.mkdir(parents=True, exist_ok=True)
            cols, rows = math.ceil(lw_ / ts), math.ceil(lh_ / ts)
            for row in range(rows):
                for col in range(cols):
                    # DZI overlap: interior edges extend `ov` px into the neighbour
                    x0 = max(col * ts - ov, 0)
                    y0 = max(row * ts - ov, 0)
                    x1 = min((col + 1) * ts + ov, lw_)
                    y1 = min((row + 1) * ts + ov, lh_)
                    cv2.imwrite(str(ldir / f"{col}_{row}.{ext}"),
                                level_img[y0:y1, x0:x1], jpg_params)
                    done += 1
                if row % 4 == 0:
                    _set_progress(key, done / max(total, 1))
            if level > 0:
                nw, nh = max(1, math.ceil(lw_ / 2)), max(1, math.ceil(lh_ / 2))
                level_img = cv2.resize(level_img, (nw, nh), interpolation=cv2.INTER_AREA)

        _meta_path(pdir).write_text(json.dumps(dict(
            width=w, height=h, tile_size=ts, overlap=ov, format=ext,
            src_mtime=image_path.stat().st_mtime,
            delete_mtime=(delete_path.stat().st_mtime
                          if delete_path and delete_path.is_file() else None),
            delete_scale=float(delete_scale),
        )))
        with _builds_lock:
            _builds.pop(key, None)  # meta.json now marks it ready
    except Exception as e:  # surface the failure to the status endpoint
        shutil.rmtree(pdir, ignore_errors=True)
        with _builds_lock:
            _builds[key] = {"state": "error", "progress": 0.0, "error": str(e)}


def dzi_descriptor(source: str, stem: str) -> str | None:
    """DZI XML for OpenSeadragon, or None if the pyramid isn't built."""
    mp = _meta_path(pyramid_dir(source, stem))
    if not mp.is_file():
        return None
    m = json.loads(mp.read_text())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
        f'Format="{m["format"]}" Overlap="{m["overlap"]}" TileSize="{m["tile_size"]}">'
        f'<Size Width="{m["width"]}" Height="{m["height"]}"/></Image>'
    )


def tile_path(source: str, stem: str, level: int, col: int, row: int) -> Path:
    return pyramid_dir(source, stem) / "files" / str(level) / f"{col}_{row}.{config.TILE_FORMAT}"
