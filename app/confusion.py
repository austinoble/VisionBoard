"""Dataset-wide Stage-2 confusion events for the interactive v9 report."""
from __future__ import annotations

from collections.abc import Callable

from shapely.geometry import Polygon

from . import annotations as ann, store


CLASSES = ["radar", "missile", "tank", "arty", "camo", "logistic", "smallvehicle", "vehicle"]
CLASS_SET = set(CLASSES)


def _poly(item: dict) -> Polygon:
    value = Polygon(item["poly"])
    return value if value.is_valid else value.buffer(0)


def _iou(left: Polygon, right: Polygon) -> float:
    if left.is_empty or right.is_empty or not left.intersects(right):
        return 0.0
    union = left.union(right).area
    return left.intersection(right).area / union if union else 0.0


def build(source: str, pcamo_mode: str,
          prediction_loader: Callable[[dict], tuple[list[dict], list[dict]]],
          iou_threshold: float = 0.30) -> dict:
    if pcamo_mode not in {"simple", "actual"}:
        raise ValueError("pcamo_mode must be simple or actual")
    events = []
    scene_count = 0
    for listed in store.list_scenes():
        if listed["source"] != source:
            continue
        scene = store.get_scene(source, listed["stem"])
        if not scene or not scene["gt_path"]:
            continue
        scene_count += 1
        predictions, regions = prediction_loader(scene)
        predictions = [item for item in predictions
                       if not item.get("in_delete") and item["cls"] in CLASS_SET]
        truth = ann.load_gt(scene["gt_path"], scene["gt_scale"])
        ann.attach_instance_ids(truth, scene["instance_id_path"], scene["instance_id_scale"])
        ann.mark_in_delete(truth, regions)
        prepared_truth = []
        for item in truth:
            if item.get("in_delete"):
                continue
            details = item.get("details") or {}
            actual = str(details.get("super_category") or item["cls"]).lower()
            parent = item["cls"] if pcamo_mode == "simple" else actual
            if parent not in CLASS_SET:
                continue
            prepared_truth.append(dict(
                item, parent=parent, actual_parent=actual,
                child=str(details.get("fine_label") or "unavailable"),
                p_camo=str(details.get("P_Camo") or "No").lower() == "yes",
                instance_id=item.get("instance_id") or "",
            ))

        image_id = next((item["instance_id"][:2] for item in prepared_truth
                         if len(item["instance_id"]) >= 2), "")
        gt_polys = [_poly(item) for item in prepared_truth]
        used = set()
        for prediction in sorted(predictions, key=lambda item: -item["score"]):
            pred_poly = _poly(prediction)
            candidates = []
            for index, gt_poly in enumerate(gt_polys):
                if index in used:
                    continue
                value = _iou(pred_poly, gt_poly)
                if value >= iou_threshold:
                    candidates.append((value, index))
            if candidates:
                value, index = max(candidates)
                used.add(index)
                target = prepared_truth[index]
                events.append(dict(
                    kind="match", source=source, image_id=image_id, image=listed["stem"],
                    instance_id=target["instance_id"], prediction_id=prediction["id"],
                    gt_parent=target["parent"], actual_parent=target["actual_parent"],
                    child=target["child"], predicted=prediction["cls"],
                    score=round(prediction["score"], 6), iou=round(value, 6),
                    p_camo=target["p_camo"],
                ))
            else:
                events.append(dict(
                    kind="fp", source=source, image_id=image_id, image=listed["stem"],
                    instance_id="", prediction_id=prediction["id"],
                    gt_parent="Background/FP", actual_parent="", child="Background/FP",
                    predicted=prediction["cls"], score=round(prediction["score"], 6),
                    iou=0.0, p_camo=False,
                ))
        for index, target in enumerate(prepared_truth):
            if index in used:
                continue
            events.append(dict(
                kind="miss", source=source, image_id=image_id, image=listed["stem"],
                instance_id=target["instance_id"], prediction_id="",
                gt_parent=target["parent"], actual_parent=target["actual_parent"],
                child=target["child"], predicted="Missed", score=None, iou=None,
                p_camo=target["p_camo"],
            ))
    return dict(classes=CLASSES, iou=iou_threshold, scene_count=scene_count,
                model="Deployed Stage-2 ensemble — shipped operating point",
                pcamo_mode=pcamo_mode, events=events)
