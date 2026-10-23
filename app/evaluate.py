"""
TP / FP / FN evaluation between prediction quads and ground-truth quads.

Standard detection matching, per class:
  1. keep predictions with score >= conf_thr
  2. sort them by score, descending
  3. each prediction greedily claims the unmatched same-class GT with highest polygon IoU,
     if that IoU >= iou_thr  -> TP (both sides matched)
  4. leftover predictions    -> FP
  5. leftover GT             -> FN

Difficult GT (DOTA flag == 1) can be ignored: predictions that match a difficult GT are
matched (so they don't count as FP) but neither side enters the stats, and unmatched
difficult GT is not an FN. This mirrors the DOTA/VOC "ignore" convention.

IoU is true polygon IoU (shapely), so rotated GT quads vs axis-aligned prediction boxes
are compared exactly, not via bounding-rect approximation.
"""
from __future__ import annotations

from shapely.geometry import Polygon


def _poly(pts: list[list[float]]) -> Polygon:
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)  # fixes self-intersecting / degenerate quads
    return p


def _iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty or not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def evaluate(preds: list[dict], gts: list[dict],
             iou_thr: float = 0.5, conf_thr: float = 0.0,
             ignore_difficult: bool = True,
             class_agnostic: bool = False) -> dict:
    """
    preds: [{id, poly, cls, score}]   gts: [{id, poly, cls, difficult}]
    Returns:
      pred_status : {pred_id: "TP" | "FP" | "below_conf"}
      gt_status   : {gt_id:   "TP" | "FN" | "ignored"}
      matches     : {pred_id: gt_id} for every matched pair (incl. difficult-ignored)
      stats       : {cls: {tp, fp, fn, precision, recall, f1}} + "_overall"
    """
    pred_status = {p["id"]: "below_conf" for p in preds}
    gt_status = {}
    matches = {}

    kept = sorted((p for p in preds if p["score"] >= conf_thr),
                  key=lambda p: -p["score"])
    for p in kept:
        pred_status[p["id"]] = "FP"  # until matched

    gt_polys = {g["id"]: _poly(g["poly"]) for g in gts}
    pred_polys = {p["id"]: _poly(p["poly"]) for p in kept}
    gts_by_cls: dict[str, list[dict]] = {}
    for g in gts:
        key = "_all" if class_agnostic else g["cls"]
        gts_by_cls.setdefault(key, []).append(g)

    claimed: set = set()
    for p in kept:
        best_iou, best_gt = 0.0, None
        key = "_all" if class_agnostic else p["cls"]
        for g in gts_by_cls.get(key, []):
            if g["id"] in claimed:
                continue
            iou = _iou(pred_polys[p["id"]], gt_polys[g["id"]])
            if iou > best_iou:
                best_iou, best_gt = iou, g
        if best_gt is not None and best_iou >= iou_thr:
            claimed.add(best_gt["id"])
            matches[p["id"]] = best_gt["id"]
            if ignore_difficult and best_gt["difficult"]:
                pred_status[p["id"]] = "TP"      # matched an ignored box: not an FP
                gt_status[best_gt["id"]] = "ignored"
            else:
                pred_status[p["id"]] = "TP"
                gt_status[best_gt["id"]] = "TP"

    for g in gts:
        if g["id"] in gt_status:
            continue
        if ignore_difficult and g["difficult"]:
            gt_status[g["id"]] = "ignored"
        else:
            gt_status[g["id"]] = "FN"

    # per-class stats (ignored boxes and their matched preds excluded)
    ignored_pred_ids = {pid for pid, gid in matches.items() if gt_status.get(gid) == "ignored"}
    gt_by_id = {g["id"]: g for g in gts}
    stats: dict[str, dict] = {}

    def bump(cls, key):
        stats.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})[key] += 1

    for p in kept:
        if p["id"] in ignored_pred_ids:
            continue
        if pred_status[p["id"]] == "TP" and class_agnostic:
            bump(gt_by_id[matches[p["id"]]]["cls"], "tp")
        else:
            bump(p["cls"], "tp" if pred_status[p["id"]] == "TP" else "fp")
    for g in gts:
        if gt_status[g["id"]] == "FN":
            bump(g["cls"], "fn")

    total = {"tp": 0, "fp": 0, "fn": 0}
    for cls, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        s["precision"] = round(tp / (tp + fp), 4) if tp + fp else None
        s["recall"] = round(tp / (tp + fn), 4) if tp + fn else None
        s["f1"] = (round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None)
        for k in total:
            total[k] += s[k]
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    total["precision"] = round(tp / (tp + fp), 4) if tp + fp else None
    total["recall"] = round(tp / (tp + fn), 4) if tp + fn else None
    total["f1"] = round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None
    stats["_overall"] = total

    # Associate wrong-class predictions with the FN they overlap. These remain an FP and
    # an FN in the metrics; the association exists only for visual review/navigation.
    error_matches = {}
    if not class_agnostic:
        error_claimed = set()
        fn_ids = {g["id"] for g in gts if gt_status.get(g["id"]) == "FN"}
        for p in kept:
            if pred_status.get(p["id"]) != "FP":
                continue
            best_iou, best_gt = 0.0, None
            for g in gts:
                if g["id"] not in fn_ids or g["id"] in error_claimed:
                    continue
                if p["cls"] == g["cls"]:
                    continue
                iou = _iou(pred_polys[p["id"]], gt_polys[g["id"]])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
            if best_gt is not None and best_iou >= iou_thr:
                error_claimed.add(best_gt["id"])
                error_matches[p["id"]] = best_gt["id"]

    return dict(pred_status=pred_status, gt_status=gt_status,
                matches=matches, error_matches=error_matches, stats=stats)
