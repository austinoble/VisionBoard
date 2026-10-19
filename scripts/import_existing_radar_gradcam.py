#!/usr/bin/env python3
"""Import the validated Test80 YOLO8m radar CAMs into the V8 explanation contract.

This is a smoke-test dataset for the UI. The complete Stage1 + two-member Stage2
generator adds/replaces entries later; this importer never deletes existing artifacts.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from shapely.geometry import Polygon


ROOT = Path("/home/jovyan/SFS/CVW/AutoRA")
SOURCE = ROOT / "experiments/stage2_y8m_radar_gradcam_test80"
DATA = ROOT / "Austine_GUI/data/lowres_full86_15oct26"
DEST = DATA / "explainability/gradcam"


def poly_iou(left, right) -> float:
    a, b = Polygon(left), Polygon(right)
    intersection = a.intersection(b).area
    union = a.area + b.area - intersection
    return float(intersection / union) if intersection > 0 and union > 0 else 0.0


def predictions(stem: str):
    path = DATA / "predictions/stage2" / f"{stem}.txt"
    out = []
    for line_index, line in enumerate(path.read_text().splitlines()):
        parts = line.split()
        if len(parts) != 10:
            continue
        points = np.asarray([float(value) for value in parts[1:9]], float).reshape(4, 2)
        out.append((line_index, int(parts[0]), points))
    return out


def main() -> None:
    manifest = json.loads((SOURCE / "run/radar_tp_manifest.json").read_text())
    entries = defaultdict(dict)
    out_cam = DEST / "cams/stage2_y8m_rgba"
    out_manifest = DEST / "manifests"
    out_cam.mkdir(parents=True, exist_ok=True)
    out_manifest.mkdir(parents=True, exist_ok=True)

    for record in manifest["true_positives"]:
        stem = record["scene"]
        nw, nh = record["normalized_size"]
        normalized = np.asarray(record["points"], float) / [nw, nh]
        candidates = [(poly_iou(normalized, points), index)
                      for index, class_index, points in predictions(stem)
                      if class_index == 4]
        best_iou, line_index = max(candidates, default=(0.0, -1))
        # The displayed box is the two-member ensemble representative, whereas the
        # source CAM run used YOLO8m alone. Their geometry is not identical, but each
        # selected member box has a unique final-box match (validated across all 71).
        if best_iou < 0.40:
            raise RuntimeError(f"{record['instance_id']}: no Stage2 line match ({best_iou:.4f})")

        image_path = DATA / "images" / f"{stem}.png"
        from PIL import Image
        width, height = Image.open(image_path).size
        scale_x, scale_y = width / nw, height / nh
        x1, y1 = record["tile_x"] * scale_x, record["tile_y"] * scale_y
        x2, y2 = (record["tile_x"] + 1024) * scale_x, (record["tile_y"] + 1024) * scale_y
        source_cam = SOURCE / "gradcam/cams" / f"{record['instance_id']}.png"
        relative = Path("cams/stage2_y8m_rgba") / f"{record['instance_id']}.png"
        target_cam = DEST / relative
        if not target_cam.exists():
            cam = cv2.imread(str(source_cam), cv2.IMREAD_GRAYSCALE)
            if cam is None:
                raise RuntimeError(f"cannot read {source_cam}")
            color = cv2.applyColorMap(cam, cv2.COLORMAP_TURBO)
            bgra = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            bgra[..., 3] = cam
            if not cv2.imwrite(str(target_cam), bgra):
                raise RuntimeError(f"cannot write {target_cam}")
        entries[stem][f"stage2:{line_index}"] = {
            "file": relative.as_posix(),
            "bounds": [x1, y1, x2, y2],
            "models": ["YOLO8m"],
            "contributors": [{
                "model": "YOLO8m", "score": record["raw_score"],
                "imgsz": record["imgsz"], "flip": record["flip"],
                "ensemble_box_iou": best_iou,
            }],
            "method": "multi-layer Grad-CAM; strongest contributing TTA view",
            "status": "validated Test80 radar TP",
        }

    for stem, predictions_by_id in entries.items():
        path = out_manifest / f"{stem}.json"
        existing = {}
        if path.exists():
            data = json.loads(path.read_text())
            existing = data.get("predictions", data)
        existing.update(predictions_by_id)
        path.write_text(json.dumps({"version": 1, "predictions": existing}, indent=2) + "\n")
    print(f"imported {sum(map(len, entries.values()))} CAMs across {len(entries)} scenes")


if __name__ == "__main__":
    main()
