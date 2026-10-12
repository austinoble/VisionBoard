"""
Adapter for an external YOLO+DINO cross-architecture ensemble.

Wraps its run_pipeline.py driver as a subprocess. The engine writes outputs/<stem>.txt in
DOTA format in the input image's own (sr2x) pixel space, which is exactly the GUI's
contract, so no post-processing is needed — we just point --out/--work-dir at the
job-private directories the GUI provides.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .base import ModelAdapter

ENGINE_ROOT = Path(os.environ.get("VISIONBOARD_DINO_ROOT", "/opt/dino_inference_model"))
DRIVER = ENGINE_ROOT / "run_pipeline.py"
PYTHON = os.environ.get("VISIONBOARD_DINO_PYTHON", sys.executable)


class DinoEnsembleAdapter(ModelAdapter):
    name = "dino_ensemble"
    display_name = "YOLO+DINO ensemble (best CV 5/6)"
    description = (
        "Cross-architecture ensemble: 4 YOLO-OBB checkpoints (+TTA) and DINO-R50 at two "
        "resolutions, all 5 folds, fused per class. INPUT MUST BE 2x SUPER-RESOLVED "
        "imagery (the space the models were trained in) — the engine does not run SR. "
        "Heavy: ~90 forward passes per 1024px tile. Output is already thresholded at the "
        "per-class operating point unless 'raw scores' is checked."
    )
    classes = ["arty", "camo", "logistic", "missile", "radar",
               "smallvehicle", "tank", "vehicle"]
    options = [
        dict(name="raw", label="Raw scores (skip operating-point thresholds)",
             type="bool", default=False,
             help="Emit every fused detection with its score; use the GUI confidence "
                  "slider to explore operating points."),
    ]

    def build_command(self, image_path: Path, out_dir: Path, work_dir: Path,
                      device: str, extra: dict) -> list[str]:
        cmd = [PYTHON, str(DRIVER),
               "--source", str(image_path),
               "--out", str(out_dir),
               "--work-dir", str(work_dir),
               "--device", device]
        if extra.get("raw"):
            cmd.append("--raw")
        return cmd
