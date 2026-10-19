#!/usr/bin/env python3
"""Generate V8 CAM overlays for every retained Stage1 prediction.

Stage1 is one class-agnostic DINOv3 ViT-H+ OBB model with a convolutional
detail branch and D4/TTA voting. The CAM target is the final fused detection
head feature, so it explains the complete DINO + convolutional decision. Each
final prediction is linked to its highest-confidence contributing TTA view.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon


ROOT = Path("/home/jovyan/SFS/CVW/AutoRA")
DATA_DEFAULT = ROOT / "Austine_GUI/data/lowres_full86_15oct26"
CHECKPOINT = ROOT / "HR_obb_train/RUN_OBB_LR/weights/lr_obb_rgb_v1_best.pt"
KIT = ROOT / "HR_obb_train/LR_KIT"
INPUT_ROOTS = [
    ROOT / "DATA/LR_Data_01sep26/test1/images",
    ROOT / "DATA/LR_Data_01sep26/test2/images",
    ROOT / "deploy_server/run_new6_15oct26/input/images",
]
TILE = 512
SCORE_FLOOR = 0.05
VIEW_NMS_IOU = 0.15
VOTE_IOU = 0.55


def polygon(points) -> Polygon:
    shape = Polygon(np.asarray(points, float).reshape(-1, 2))
    return shape if shape.is_valid else shape.buffer(0)


def polygon_iou(left: Polygon, right: Polygon) -> float:
    intersection = left.intersection(right).area
    union = left.area + right.area - intersection
    return float(intersection / union) if intersection > 0 and union > 0 else 0.0


def input_lookup() -> dict[str, Path]:
    images = {}
    for root in INPUT_ROOTS:
        for suffix in ("*.tif", "*.tiff"):
            for path in root.glob(suffix):
                images[path.stem] = path
    return images


def load_targets(data: Path, stem: str, width: int, height: int) -> list[dict]:
    targets = []
    path = data / "predictions/stage1" / f"{stem}.txt"
    for line_index, line in enumerate(path.read_text().splitlines()):
        fields = line.split()
        if len(fields) != 10:
            continue
        points = np.asarray([float(value) for value in fields[1:9]], float).reshape(4, 2)
        points *= [width, height]
        targets.append({
            "id": f"stage1:{line_index}", "points": points,
            "polygon": polygon(points), "display_score": float(fields[9]),
        })
    return targets


def decode_tile_meta(output: dict, score_floor: float, stride: int, topk: int = 500):
    """Deployed dense-head decode plus the source cell needed for Grad-CAM."""
    from obb.decode import _nms_pool

    center = torch.sigmoid(output["center"])
    peaks = _nms_pool(center)[0, 0]
    height, width = peaks.shape
    flat = peaks.reshape(-1)
    count = int((flat > score_floor).sum().item())
    if not count:
        return np.zeros((0, 5)), np.zeros(0), np.zeros((0, 2), int)
    scores, indices = torch.topk(flat, min(topk, count))
    ys, xs = (indices // width).long(), (indices % width).long()
    offset = output["offset"][0, :, ys, xs]
    size = output["size"][0, :, ys, xs]
    angle = F.normalize(output["angle"][0, :, ys, xs], dim=0)
    cx = (xs.float() + offset[0]) * stride
    cy = (ys.float() + offset[1]) * stride
    box_width, box_height = torch.exp(size[0]), torch.exp(size[1])
    theta = torch.atan2(angle[1], angle[0]) * 0.5
    boxes = torch.stack([cx, cy, box_width, box_height, theta], -1).cpu().numpy()
    cells = torch.stack([xs, ys], -1).cpu().numpy()
    return boxes, scores.cpu().numpy(), cells


@torch.no_grad()
def infer_view(model, image: np.ndarray, device: str, view_index: int) -> list[dict]:
    from obb import geometry as geometry
    from obb.tiler import grid_tiles

    stride = int(model.out_stride)
    height, width = image.shape[:2]
    origins = grid_tiles(height, width, TILE)
    all_quads, all_scores, all_meta = [], [], []
    batch_images, batch_origins = [], []

    def flush() -> None:
        if not batch_images:
            return
        tensor = torch.from_numpy(np.stack(batch_images)).permute(0, 3, 1, 2).float().to(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            output = model(tensor)
        output = {key: value.float() for key, value in output.items()}
        for batch_index, (x0, y0) in enumerate(batch_origins):
            single = {key: value[batch_index:batch_index + 1] for key, value in output.items()}
            boxes, scores, cells = decode_tile_meta(single, SCORE_FLOOR, stride)
            if not len(boxes):
                continue
            quads = geometry.rboxes_to_quads(boxes)
            quads[..., 0] += x0
            quads[..., 1] += y0
            all_quads.append(quads)
            all_scores.append(scores)
            all_meta.extend({
                "view_index": view_index, "tile_x": x0, "tile_y": y0,
                "cell_x": int(cell[0]), "cell_y": int(cell[1]),
                "raw_score": float(score),
            } for cell, score in zip(cells, scores))
        batch_images.clear()
        batch_origins.clear()

    channels = image.shape[2]
    for x0, y0 in origins:
        crop = np.zeros((TILE, TILE, channels), np.float32)
        source = image[y0:y0 + TILE, x0:x0 + TILE]
        crop[:source.shape[0], :source.shape[1]] = source
        batch_images.append(crop)
        batch_origins.append((x0, y0))
        if len(batch_images) == 8:
            flush()
    flush()
    if not all_quads:
        return []
    quads = np.concatenate(all_quads)
    scores = np.concatenate(all_scores)
    keep = geometry.rotated_nms(quads, scores, thr=VIEW_NMS_IOU)
    return [{**all_meta[index], "view_points": quads[index],
             "view_polygon": polygon(quads[index])} for index in keep]


def provenance(model, rgb: np.ndarray, device: str) -> list[dict]:
    from obb.decode import _d4_views

    native_height, native_width = rgb.shape[:2]
    records = []
    for view_index, (view, inverse) in enumerate(_d4_views(rgb)):
        for record in infer_view(model, view, device, view_index):
            native_points = inverse(record["view_points"], native_height, native_width)
            records.append({**record, "native_points": native_points,
                            "native_polygon": polygon(native_points)})
    return records


def strongest_contributor(target: dict, records: list[dict]) -> tuple[dict | None, float]:
    values = [(polygon_iou(target["polygon"], record["native_polygon"]), record)
              for record in records]
    best_iou, best = max(values, default=(0.0, None), key=lambda pair: pair[0])
    if best is None or best_iou < 0.30:
        return None, best_iou
    cluster = [(record["raw_score"], record) for value, record in values if value >= VOTE_IOU]
    return (max(cluster, default=(best["raw_score"], best), key=lambda pair: pair[0])[1], best_iou)


def tile_for_record(rgb: np.ndarray, record: dict) -> tuple[np.ndarray, int, int, bool, int]:
    from obb.decode import _d4_views

    view_index = record["view_index"]
    view = _d4_views(rgb)[view_index][0]
    x0, y0 = record["tile_x"], record["tile_y"]
    crop = np.zeros((TILE, TILE, 3), np.float32)
    source = view[y0:y0 + TILE, x0:x0 + TILE]
    crop[:source.shape[0], :source.shape[1]] = source
    rotation, flipped = view_index // 2, bool(view_index % 2)
    return crop, rotation, flipped, view.shape[1], view.shape[0]


def gradcam(model, rgb: np.ndarray, record: dict) -> tuple[np.ndarray, list[float]]:
    """Memory-efficient Grad-CAM: detach at final fused head feature."""
    crop, rotation, flipped, view_width, _view_height = tile_for_record(rgb, record)
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)[None]).float().cuda()
    activation = {}

    def capture(_module, _inputs, output):
        # All upstream parameters are frozen. Turning this fused feature into a leaf
        # computes exactly the gradient needed for CAM without retaining ViT-H activations.
        output.requires_grad_(True)
        output.retain_grad()
        activation["value"] = output

    model.requires_grad_(False)
    handle = model.head.share[-1].register_forward_hook(capture)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    try:
        with torch.enable_grad(), torch.autocast("cuda", dtype=dtype):
            output = model(tensor)
            score = output["center"][0, 0, record["cell_y"], record["cell_x"]]
        model.zero_grad(set_to_none=True)
        score.backward()
        feature = activation["value"].detach().float()[0]
        gradient = activation["value"].grad.detach().float()[0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * feature).sum(dim=0)).cpu().numpy()
    finally:
        handle.remove()
    cam = cv2.resize(cam, (TILE, TILE), interpolation=cv2.INTER_LINEAR)
    maximum = float(cam.max())
    if maximum > 0:
        cam /= maximum

    # Undo the view flip/rotation so the CAM is upright in native scene space.
    x0, y0 = record["tile_x"], record["tile_y"]
    if flipped:
        cam = cv2.flip(cam, 1)
        x0 = view_width - (x0 + TILE)
    cam = np.rot90(cam, -rotation).copy()
    native_height, native_width = rgb.shape[:2]
    corners = np.asarray([[record["tile_x"], record["tile_y"]],
                          [record["tile_x"] + TILE, record["tile_y"]],
                          [record["tile_x"] + TILE, record["tile_y"] + TILE],
                          [record["tile_x"], record["tile_y"] + TILE]], float)
    from obb.decode import _d4_views
    inverse = _d4_views(rgb)[record["view_index"]][1]
    native = inverse(corners, native_height, native_width)
    x1, y1 = native.min(axis=0)
    x2, y2 = native.max(axis=0)
    del tensor, output
    torch.cuda.empty_cache()
    return np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0), [x1, y1, x2, y2]


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


def load_model(device: str):
    from obb.model import OBBNet

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = OBBNet(checkpoint["backbone_meta"], out_stride=checkpoint.get("out_stride", 2),
                   in_ch=checkpoint.get("in_ch", 3), n_cls=checkpoint.get("n_cls", 0),
                   n_empty=checkpoint.get("n_empty", 0)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument(
        "--scene-list", type=Path,
        help="Optional text file containing one exact scene stem per line.",
    )
    args = parser.parse_args()
    os.environ.setdefault("OBB_WORK_ROOT", "/tmp/visionboard_v8_gradcam")
    os.environ.setdefault("S1_WORK_ROOT", "/tmp/visionboard_v8_gradcam")
    sys.path.insert(0, str(KIT))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    from stage1 import imagery
    images = input_lookup()
    stems = sorted(path.stem for path in (args.data / "images").glob("*.png"))
    if args.scene_list:
        requested = [line.strip() for line in args.scene_list.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        available = set(stems)
        unknown = [stem for stem in requested if stem not in available]
        if unknown:
            raise SystemExit(f"scene list contains {len(unknown)} unknown stems: {unknown[:3]}")
        stems = requested
    if args.limit_scenes:
        stems = stems[:args.limit_scenes]
    missing = [stem for stem in stems if stem not in images]
    if missing:
        raise SystemExit(f"missing native TIFFs for {len(missing)} scenes: {missing[:3]}")
    model = load_model("cuda")
    grad_root = args.data / "explainability/gradcam"
    complete = failed = 0

    for scene_index, stem in enumerate(stems, 1):
        started = time.time()
        rgb, native_height, native_width = imagery.read_full_rgb(str(images[stem]))
        targets = load_targets(args.data, stem, native_width, native_height)
        records = provenance(model, rgb, "cuda")
        from PIL import Image
        display_width, display_height = Image.open(args.data / "images" / f"{stem}.png").size
        entries = {}
        for target in targets:
            record, match_iou = strongest_contributor(target, records)
            if record is None:
                failed += 1
                print(f"  !! {stem} {target['id']}: no TTA contributor (best IoU={match_iou:.3f})",
                      flush=True)
                continue
            cam, bounds = gradcam(model, rgb, record)
            safe_id = target["id"].replace(":", "_")
            relative = Path("cams/stage1") / f"{stem}__{safe_id}.png"
            save_rgba(grad_root / relative, cam)
            x1, y1, x2, y2 = bounds
            entries[target["id"]] = {
                "file": relative.as_posix(),
                "bounds": [x1 * display_width / native_width, y1 * display_height / native_height,
                           x2 * display_width / native_width, y2 * display_height / native_height],
                "models": ["DINOv3 OBB"],
                "contributors": [{
                    "model": "DINOv3 OBB", "score": record["raw_score"],
                    "view": record["view_index"], "rotation": record["view_index"] // 2,
                    "flip": bool(record["view_index"] % 2),
                    "tile_x": record["tile_x"], "tile_y": record["tile_y"],
                    "match_iou": match_iou,
                }],
                "method": "fused-head Grad-CAM; strongest contributing D4/TTA view",
                "status": "retained deployed Stage1 prediction",
            }
            complete += 1
        write_manifest(grad_root / "manifests" / f"{stem}.json", entries)
        print(f"[{scene_index:02d}/{len(stems)}] {stem[:45]:45s} "
              f"CAM {len(entries)}/{len(targets)}  {time.time()-started:.1f}s", flush=True)
    print(f"DONE Stage1 CAM={complete}, failed={failed}, scenes={len(stems)}")


if __name__ == "__main__":
    main()
