"""
DOTA-style annotation file parsing.

One box per line:  x1 y1 x2 y2 x3 y3 x4 y4 class_name trailing
  - prediction files: trailing = confidence (float)
  - ground-truth files: trailing = difficulty flag (0/1), per DOTA convention

Both HBB (axis-aligned) and OBB (rotated quad) boxes are supported — the GUI draws the
4-corner polygon exactly as stored, so any quad renders correctly.

All coordinates returned by this module are ALREADY scaled into displayed-image pixel
space (the `scale` argument applies the source's pred_scale / gt_scale).
"""
from __future__ import annotations

from pathlib import Path

from . import config


STAGE2_VECTOR_CLASSES = [
    "arty", "camo", "logistic", "missile",
    "radar", "smallvehicle", "tank", "vehicle",
]

DETAILED_FIELDS = [
    "super_category", "fine_label", "Normal", "Snow", "Dark", "Quality",
    "Functionality", "Position", "P_Camo", "P_Cut", "Remarks", "User",
    "Is_Empty(ML)", "SOC", "Camo Instance", "Background Matching", "difficult",
]


def _parse_line(line: str) -> tuple[list[list[float]], str, float] | None:
    parts = line.split()
    if len(parts) < 9:
        return None
    try:
        coords = [float(v) for v in parts[:8]]
    except ValueError:
        return None
    cls = parts[8]
    trailing = 0.0
    if len(parts) > 9:
        try:
            trailing = float(parts[9])
        except ValueError:
            trailing = 0.0
    poly = [[coords[0], coords[1]], [coords[2], coords[3]],
            [coords[4], coords[5]], [coords[6], coords[7]]]
    return poly, cls, trailing


def _class_name(token: str, class_names: list[str] | None, default: str = "object") -> str:
    if class_names is None:
        return default
    try:
        index = int(token)
    except ValueError:
        return token
    return class_names[index] if 0 <= index < len(class_names) else f"class_{index}"


def load_preds(path: Path, scale: float = 1.0, *, fmt: str = "dota",
               width: int | None = None, height: int | None = None,
               class_names: list[str] | None = None, stage: str = "pred") -> list[dict]:
    """[{id, poly, cls, score[, raw]}] — poly in displayed-image pixels, score in 0..1.

    Fused/composite scores may exceed 1.0 with class-dependent scales, so scores are
    normalized per config.NORMALIZE_SCORES (see there). When a score was rescaled, the
    original is kept under 'raw'.
    """
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        if fmt == "lowres_normalized":
            parts = line.split()
            if len(parts) != 10 or width is None or height is None:
                continue
            try:
                coords = [float(v) for v in parts[1:9]]
                score = float(parts[9])
            except ValueError:
                continue
            poly = [[coords[j] * width, coords[j + 1] * height]
                    for j in range(0, 8, 2)]
            cls = _class_name(parts[0], class_names)
        else:
            parsed = _parse_line(line.strip())
            if not parsed:
                continue
            poly, cls, score = parsed
            poly = [[x * scale, y * scale] for x, y in poly]
        out.append(dict(
            id=f"{stage}:{i}",
            layer=stage,
            poly=poly,
            cls=cls,
            score=round(score, 4),
        ))
    _normalize_scores(out)
    return out


def load_prediction_vectors(path: Path | None, scale: float = 1.0) -> list[dict]:
    """Load Stage-2 polygons followed by the full eight-class score vector."""
    if path is None or not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 10 + len(STAGE2_VECTOR_CLASSES):
            continue
        try:
            coords = [float(v) * scale for v in parts[:8]]
            scores = [float(v) for v in parts[10:10 + len(STAGE2_VECTOR_CLASSES)]]
        except ValueError:
            continue
        out.append({
            "poly": [[coords[i], coords[i + 1]] for i in range(0, 8, 2)],
            "scores": dict(zip(STAGE2_VECTOR_CLASSES, scores)),
        })
    return out


def attach_top_classes(preds: list[dict], vectors: list[dict],
                       min_iou: float = 0.20) -> int:
    """Attach the best-overlapping vector's three strongest classes to predictions."""
    if not preds or not vectors:
        return 0
    from shapely.geometry import Polygon

    prepared = []
    for item in vectors:
        poly = Polygon(item["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            prepared.append((poly, poly.bounds, item["scores"]))

    attached = 0
    for item in preds:
        poly = Polygon(item["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        ax1, ay1, ax2, ay2 = poly.bounds
        best_iou, best_scores = 0.0, None
        for other, (bx1, by1, bx2, by2), scores in prepared:
            if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
                continue
            intersection = poly.intersection(other).area
            if intersection <= 0:
                continue
            union = poly.area + other.area - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_scores = iou, scores
        if best_scores is not None and best_iou >= min_iou:
            ranked = sorted(best_scores.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
            item["top_classes"] = [
                {"cls": cls, "score": round(float(score), 4)} for cls, score in ranked
            ]
            item["vector_iou"] = round(best_iou, 4)
            attached += 1
    return attached


def suppress_stage1(stage1: list[dict], stage2: list[dict], iou_threshold: float) -> int:
    """Mark Stage-1 boxes replaced by a highly-overlapping Stage-2 box."""
    if not stage1 or not stage2:
        return 0
    from shapely.geometry import Polygon

    stage2_polys = []
    for item in stage2:
        if item.get("in_delete"):
            continue
        poly = Polygon(item["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            stage2_polys.append(poly)
    count = 0
    for item in stage1:
        if item.get("in_delete"):
            continue
        poly = Polygon(item["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        for other in stage2_polys:
            if not poly.intersects(other):
                continue
            intersection = poly.intersection(other).area
            union = poly.area + other.area - intersection
            if union > 0 and intersection / union > iou_threshold:
                item["suppressed_by_stage2"] = True
                count += 1
                break
    return count


def threshold_and_dedup(preds: list[dict], thresholds: dict[str, float],
                        priority: list[str], iou_threshold: float) -> list[dict]:
    """Apply the shipped Stage-2 confidence gates and priority cross-class dedup."""
    if not preds:
        return []
    from shapely.geometry import Polygon

    priority_set = set(priority)
    candidates = []
    for item in preds:
        if item["score"] < float(thresholds.get(item["cls"], 1.01)):
            continue
        poly = Polygon(item["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        candidates.append((0 if item["cls"] in priority_set else 1,
                           -item["score"], item, poly))
    candidates.sort(key=lambda row: (row[0], row[1]))

    kept: list[tuple[dict, Polygon]] = []
    for _, _, item, poly in candidates:
        duplicate = False
        for _, winner in kept:
            if not poly.intersects(winner):
                continue
            inter = poly.intersection(winner).area
            union = poly.area + winner.area - inter
            if union > 0 and inter / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((item, poly))
    return [item for item, _ in kept]


def group_display_classes(items: list[dict]) -> None:
    """Collapse Stage-2/GT classes to radar, missile, or other for this GUI version."""
    for item in items:
        original = item["cls"]
        if original not in {"radar", "missile"}:
            item["raw_cls"] = original
            item["cls"] = "other"


def _normalize_scores(preds: list[dict]) -> None:
    """Map scores into 0..1 in place — only groups whose max exceeds 1.0 are rescaled,
    so ordinary probability scores pass through unchanged."""
    mode = config.NORMALIZE_SCORES
    if not mode or not preds:
        return
    groups: dict[str, list[dict]] = {}
    for p in preds:
        groups.setdefault(p["cls"] if mode == "per_class" else "_all", []).append(p)
    for members in groups.values():
        top = max(p["score"] for p in members)
        if top <= 1.0:
            continue
        for p in members:
            p["raw"] = p["score"]
            p["score"] = round(p["score"] / top, 4)


def load_delete_regions(path: Path, scale: float = 1.0) -> list[dict]:
    """[{id, poly}] — delete-region polygons in displayed-image pixels.

    Format: one polygon per line, any number of vertices, coordinates first and a label
    (e.g. "Delete") last:  x1 y1 x2 y2 ... xN yN Delete
    """
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        floats = []
        for tok in line.split():
            try:
                floats.append(float(tok))
            except ValueError:
                break  # reached the label
        if len(floats) < 6:      # need at least 3 points
            continue
        if len(floats) % 2:
            floats = floats[:-1]
        out.append(dict(
            id=i,
            poly=[[floats[j] * scale, floats[j + 1] * scale]
                  for j in range(0, len(floats), 2)],
        ))
    return out


def mark_in_delete(preds: list[dict], regions: list[dict]) -> int:
    """Flag predictions whose centre lies inside any delete region (in_delete=True).

    The GUI hides flagged boxes and evaluation skips them — detections in areas the
    annotators voided are neither shown nor counted as FP. Returns how many were flagged.
    """
    if not preds or not regions:
        return 0
    from shapely.geometry import Polygon
    from shapely.prepared import prep

    region_polys = []
    for r in regions:
        poly = Polygon(r["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)  # delete outlines are hand-drawn and often messy
        if not poly.is_empty:
            region_polys.append(prep(poly))
    n = 0
    for p in preds:
        centre = Polygon(p["poly"]).centroid
        if any(rp.contains(centre) for rp in region_polys):
            p["in_delete"] = True
            n += 1
    return n


def load_gt(path: Path, scale: float = 1.0) -> list[dict]:
    """[{id, poly, cls, difficult}] — poly in displayed-image pixels."""
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        parsed = _parse_line(line.strip())
        if not parsed:
            continue
        poly, cls, difficult = parsed
        out.append(dict(
            id=i,
            poly=[[x * scale, y * scale] for x, y in poly],
            cls=cls,
            difficult=int(difficult),
        ))
    return out


def attach_instance_ids(gt: list[dict], path: Path | None, scale: float = 1.0,
                        min_iou: float = 0.5) -> int:
    """Attach IDs and all labels_detailed fields to labels_simple objects by polygon IoU.

    Detailed records begin with the same eight coordinates and class as DOTA GT, contain
    additional metadata, and store the stable instance ID in their final column. Matching
    by scaled polygon geometry avoids relying on line order.
    """
    if not gt or path is None or not path.is_file():
        return 0
    from shapely.geometry import Polygon

    detailed = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 26:
            continue
        try:
            coords = [float(v) for v in parts[:8]]
        except ValueError:
            continue
        poly = Polygon([(coords[i] * scale, coords[i + 1] * scale)
                        for i in range(0, 8, 2)])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            detail = dict(zip(DETAILED_FIELDS, parts[8:25]))
            detail["instance_id"] = parts[25]
            detail["coordinates"] = [float(v) for v in parts[:8]]
            detailed.append((poly, detail))

    used = set()
    attached = 0
    for item in gt:
        poly = Polygon(item["poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        best_iou, best_index = 0.0, None
        for index, (other, _detail) in enumerate(detailed):
            # labels_simple may intentionally reclassify an object (for example, a
            # detailed missile record can become camo under the simplified rules), so
            # stable IDs are joined by geometry rather than by the pre-rule class name.
            if index in used or not poly.intersects(other):
                continue
            intersection = poly.intersection(other).area
            union = poly.area + other.area - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= min_iou:
            used.add(best_index)
            item["details"] = detailed[best_index][1]
            item["instance_id"] = item["details"]["instance_id"]
            attached += 1
    return attached
