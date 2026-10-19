#!/usr/bin/env python3
"""Generate V8 CAM overlays for every retained Stage2 ensemble prediction.

The deployed Stage2 score is a non-differentiable noisy-or fusion of two YOLO OBB
models and four TTA views per model. This script reproduces the raw detections,
attributes every final retained box to its IoU fusion cluster, and computes one
Grad-CAM from the single highest-confidence contributing model/TTA view.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from shapely.geometry import Polygon


ROOT = Path("/home/jovyan/SFS/CVW/AutoRA")
PKG = ROOT / "HR_obb_train/stage2_sep_w2"
DATA_DEFAULT = ROOT / "Austine_GUI/data/lowres_full86_15oct26"
INPUT_ROOTS = [
    ROOT / "deploy_server/cmp3/in/native",
    ROOT / "deploy_server/run_new6_15oct26/work/stage2_in",
]
VENDOR = ROOT / "experiments/stage2_y8m_radar_gradcam_test80/vendor"
CLASSES = ["arty", "camo", "logistic", "missile", "radar", "smallvehicle", "tank", "vehicle"]
# Deployed fixed-format files retain the historical ten-slot class numbering
# (radome=5 and spa=7 are unused by these eight-class ensemble weights).
DEPLOY_TO_MODEL_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 8: 6, 9: 7}
TARGET_GSD = 0.30
TILE = 1024
OVERLAP = 0.20
CONTEXT = 512
MODEL_SPECS = [
    ("YOLO8m", "prod_y8m_best.pt", [15, 18, 21]),
    ("YOLO11l", "prod_y11l_best.pt", [16, 19, 22]),
]


def polygon(points) -> Polygon:
    shape = Polygon(np.asarray(points, float).reshape(-1, 2))
    return shape if shape.is_valid else shape.buffer(0)


def polygon_iou(left: Polygon, right: Polygon) -> float:
    intersection = left.intersection(right).area
    union = left.area + right.area - intersection
    return float(intersection / union) if intersection > 0 and union > 0 else 0.0


def starts(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, step))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def input_lookup() -> tuple[dict[str, Path], dict[str, float]]:
    images, gsd = {}, {}
    for root in INPUT_ROOTS:
        for path in root.glob("*.png"):
            images[path.stem] = path
        table = root / "gsd.csv"
        if table.is_file():
            for row in csv.DictReader(table.open()):
                gsd[Path(row["filename"]).stem] = float(row["gsd"])
    return images, gsd


def load_targets(data: Path, stem: str, width: int, height: int) -> list[dict]:
    targets = []
    path = data / "predictions/stage2" / f"{stem}.txt"
    for line_index, line in enumerate(path.read_text().splitlines()):
        fields = line.split()
        if len(fields) != 10:
            continue
        deployed_class_index = int(fields[0])
        if deployed_class_index not in DEPLOY_TO_MODEL_CLASS:
            continue
        class_index = DEPLOY_TO_MODEL_CLASS[deployed_class_index]
        points = np.asarray([float(value) for value in fields[1:9]], float).reshape(4, 2)
        points *= [width, height]
        targets.append({
            "id": f"stage2:{line_index}", "class_index": class_index,
            "class": CLASSES[class_index], "points": points,
            "polygon": polygon(points), "display_score": float(fields[9]),
        })
    return targets


def scene_records(image: np.ndarray, models: list[dict], scales: list[int], flips: list[str],
                  floor: float) -> dict[int, list[dict]]:
    """All post-NMS member detections with exact tile/view provenance."""
    height, width = image.shape[:2]
    step = int(TILE * (1 - OVERLAP))
    records = collections.defaultdict(list)
    for y0 in starts(height, TILE, step):
        for x0 in starts(width, TILE, step):
            crop = image[y0:y0 + TILE, x0:x0 + TILE]
            if crop.shape[:2] != (TILE, TILE):
                padded = np.zeros((TILE, TILE, 3), np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            for model_index, spec in enumerate(models):
                for image_size in scales:
                    for flip in flips:
                        view = crop if flip == "none" else cv2.flip(crop, 1)
                        result = spec["yolo"].predict(
                            view, conf=floor, imgsz=image_size, max_det=400,
                            verbose=False, device=0)[0]
                        if result.obb is None or not len(result.obb):
                            continue
                        points = result.obb.xyxyxyxy.detach().cpu().numpy()
                        scores = result.obb.conf.detach().cpu().numpy()
                        class_ids = result.obb.cls.detach().cpu().numpy().astype(int)
                        for view_points, score, class_index in zip(points, scores, class_ids):
                            if not 0 <= class_index < len(CLASSES):
                                continue
                            scene_points = view_points.copy()
                            if flip == "h":
                                scene_points[..., 0] = (TILE - 1) - scene_points[..., 0]
                            scene_points += [x0, y0]
                            records[class_index].append({
                                "model_index": model_index, "model": spec["name"],
                                "raw_score": float(score), "points": scene_points,
                                "polygon": polygon(scene_points), "view_points": view_points,
                                "tile_x": x0, "tile_y": y0,
                                "imgsz": image_size, "flip": flip,
                            })
    return records


class PredictionTensor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        output = self.model(tensor)
        return output[0] if isinstance(output, tuple) else output


class AnchorClassTarget:
    def __init__(self, anchor: int, class_index: int):
        self.anchor = anchor
        self.class_index = class_index

    def __call__(self, output):
        return output[4 + self.class_index, self.anchor]


def input_tensor(view: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(view, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
    return tensor.unsqueeze(0).cuda().requires_grad_(True)


def select_anchor(decoded: torch.Tensor, record: dict, class_index: int) -> tuple[int, float, float]:
    from ultralytics.utils.ops import xywhr2xyxyxyxy

    image_size = int(record["imgsz"])
    xywhr = torch.cat((decoded[0, :4, :].T, decoded[0, 12:13, :].T), dim=1)
    points = xywhr2xyxyxyxy(xywhr).detach().cpu().numpy()
    scores = decoded[0, 4 + class_index].detach().cpu().numpy()
    target = polygon(np.asarray(record["view_points"], float) * (image_size / TILE))
    candidates = np.flatnonzero(scores >= 0.01)
    if len(candidates) > 1500:
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")[:1500]]
    values = [(polygon_iou(target, polygon(points[index])), int(index)) for index in candidates]
    best_iou, best_index = max(values, default=(-1.0, -1))
    if best_index < 0 or best_iou < 0.45:
        raise RuntimeError(f"raw-anchor link failed: best IoU {best_iou:.4f}")
    return best_index, float(scores[best_index]), best_iou


def crop_view(image: np.ndarray, record: dict) -> np.ndarray:
    x0, y0 = int(record["tile_x"]), int(record["tile_y"])
    crop = image[y0:y0 + TILE, x0:x0 + TILE]
    if crop.shape[:2] != (TILE, TILE):
        padded = np.zeros((TILE, TILE, 3), np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop if record["flip"] == "none" else cv2.flip(crop, 1)


def member_cam(image: np.ndarray, spec: dict, record: dict, class_index: int) -> tuple[np.ndarray, dict]:
    from pytorch_grad_cam import GradCAM

    network = spec["network"]
    network.requires_grad_(True)
    # ``YOLO.predict`` runs under torch.inference_mode and may refresh the OBB
    # head's cached anchors/strides with inference tensors when image size
    # changes.  Grad-CAM's differentiable forward cannot save those tensors for
    # backward, so materialize ordinary clones before the gradient pass.
    head = network.model[-1]
    for attribute in ("anchors", "strides"):
        value = getattr(head, attribute, None)
        if isinstance(value, torch.Tensor):
            setattr(head, attribute, value.clone())
    wrapper = PredictionTensor(network)
    view = crop_view(image, record)
    tensor = input_tensor(view, int(record["imgsz"]))
    with torch.enable_grad():
        decoded = wrapper(tensor)
        anchor, anchor_score, anchor_iou = select_anchor(decoded, record, class_index)
    target_layers = [network.model[index] for index in spec["layers"]]
    network.zero_grad(set_to_none=True)
    with GradCAM(model=wrapper, target_layers=target_layers) as gradcam:
        cam = gradcam(input_tensor=tensor,
                      targets=[AnchorClassTarget(anchor, class_index)])[0]
    cam = cv2.resize(cam, (TILE, TILE), interpolation=cv2.INTER_LINEAR)
    if record["flip"] == "h":
        cam = cv2.flip(cam, 1)
    del tensor, decoded
    torch.cuda.empty_cache()
    return np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0), {
        "model": spec["name"], "score": record["raw_score"],
        "anchor_score": anchor_score, "anchor_iou": anchor_iou,
        "imgsz": record["imgsz"], "flip": record["flip"],
        "tile_x": record["tile_x"], "tile_y": record["tile_y"],
    }


def context_bounds(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy = points.mean(axis=0)
    side = min(CONTEXT, width, height)
    x1 = int(np.clip(round(cx - side / 2), 0, max(width - side, 0)))
    y1 = int(np.clip(round(cy - side / 2), 0, max(height - side, 0)))
    return x1, y1, x1 + side, y1 + side


def stitch(members: list[tuple[np.ndarray, dict]], bounds) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    height, width = y2 - y1, x2 - x1
    total = np.zeros((height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    for cam, metadata in members:
        tx1, ty1 = metadata["tile_x"], metadata["tile_y"]
        tx2, ty2 = tx1 + TILE, ty1 + TILE
        ix1, iy1, ix2, iy2 = max(x1, tx1), max(y1, ty1), min(x2, tx2), min(y2, ty2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue
        weight = max(float(metadata["score"]), 1e-6)
        target = np.s_[iy1-y1:iy2-y1, ix1-x1:ix2-x1]
        source = np.s_[iy1-ty1:iy2-ty1, ix1-tx1:ix2-tx1]
        total[target] += cam[source] * weight
        weights[target] += weight
    output = np.divide(total, weights, out=np.zeros_like(total), where=weights > 0)
    maximum = float(output.max())
    return output / maximum if maximum > 0 else output


def save_rgba(path: Path, cam: np.ndarray) -> None:
    values = np.uint8(np.clip(cam, 0, 1) * 255)
    color = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    bgra = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
    # Keep the complete context patch visible. CAM magnitude is encoded by the
    # Turbo colour scale; the GUI's opacity control governs transparency.
    bgra[..., 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgra):
        raise RuntimeError(f"could not write {path}")


def write_manifest(path: Path, additions: dict) -> None:
    existing = {}
    if path.exists():
        data = json.loads(path.read_text())
        existing = data.get("predictions", data)
    existing.update(additions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "predictions": existing}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument(
        "--scene-list", type=Path,
        help="Optional text file containing one exact scene stem per line.",
    )
    args = parser.parse_args()
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    images, gsd_by_stem = input_lookup()
    display_stems = sorted(path.stem for path in (args.data / "images").glob("*.png"))
    if args.scene_list:
        requested = [line.strip() for line in args.scene_list.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        available = set(display_stems)
        unknown = [stem for stem in requested if stem not in available]
        if unknown:
            raise SystemExit(f"scene list contains {len(unknown)} unknown stems: {unknown[:3]}")
        display_stems = requested
    if args.limit_scenes:
        display_stems = display_stems[:args.limit_scenes]
    missing = [stem for stem in display_stems if stem not in images or stem not in gsd_by_stem]
    if missing:
        raise SystemExit(f"missing Stage2 inputs/GSD for {len(missing)} scenes: {missing[:3]}")

    config = yaml.safe_load((PKG / "config.yaml").read_text())
    scales = [int(value) for value in config["tta"]["scales"]]
    flips = list(config["tta"]["flips"])
    floor = float(config["save_floor"])
    fusion_iou = float(config["nms_iou"])

    from ultralytics import YOLO
    models = []
    for name, filename, layers in MODEL_SPECS:
        yolo = YOLO(str(PKG / "weights" / filename))
        models.append({"name": name, "yolo": yolo,
                       "network": yolo.model.cuda().eval(), "layers": layers})

    grad_root = args.data / "explainability/gradcam"
    complete = failed = 0
    for scene_index, stem in enumerate(display_stems, 1):
        start = time.time()
        image0 = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        scale = gsd_by_stem[stem] / TARGET_GSD
        image = image0 if abs(scale - 1.0) < 1e-6 else cv2.resize(
            image0, (max(round(image0.shape[1] * scale), 1),
                     max(round(image0.shape[0] * scale), 1)), interpolation=cv2.INTER_LINEAR)
        height, width = image.shape[:2]
        targets = load_targets(args.data, stem, width, height)
        raw = scene_records(image, models, scales, flips, floor)
        from PIL import Image
        display_width, display_height = Image.open(args.data / "images" / f"{stem}.png").size
        entries = {}

        for target in targets:
            candidates = [record for record in raw[target["class_index"]]
                          if polygon_iou(target["polygon"], record["polygon"]) >= fusion_iou]
            if not candidates:
                best = sorted(((polygon_iou(target["polygon"], record["polygon"]), record)
                               for record in raw[target["class_index"]]), reverse=True,
                              key=lambda pair: pair[0])
                failed += 1
                print(f"  !! {stem} {target['id']}: no fusion contributor; "
                      f"best IoU={best[0][0] if best else 0:.3f}", flush=True)
                continue
            representative = max(candidates, key=lambda record: record["raw_score"])
            bounds = context_bounds(target["points"], width, height)
            try:
                cam, meta = member_cam(
                    image, models[representative["model_index"]], representative,
                    target["class_index"],
                )
            except Exception as exc:
                print(f"  !! {stem} {target['id']} {representative['model']}: "
                      f"Grad-CAM failed: {exc}", flush=True)
                failed += 1
                continue
            selected = stitch([(cam, meta)], bounds)
            safe_id = target["id"].replace(":", "_")
            relative = Path("cams/stage2") / f"{stem}__{safe_id}.png"
            save_rgba(grad_root / relative, selected)
            x1, y1, x2, y2 = bounds
            entries[target["id"]] = {
                "file": relative.as_posix(),
                "bounds": [x1 * display_width / width, y1 * display_height / height,
                           x2 * display_width / width, y2 * display_height / height],
                "models": [meta["model"]],
                "contributors": [meta],
                "method": "single highest-confidence ensemble contributor CAM",
                "status": "retained deployed Stage2 prediction",
            }
            complete += 1
        write_manifest(grad_root / "manifests" / f"{stem}.json", entries)
        print(f"[{scene_index:02d}/{len(display_stems)}] {stem[:45]:45s} "
              f"CAM {len(entries)}/{len(targets)}  {time.time()-start:.1f}s", flush=True)
    print(f"DONE Stage2 CAM={complete}, failed={failed}, scenes={len(display_stems)}")


if __name__ == "__main__":
    main()
