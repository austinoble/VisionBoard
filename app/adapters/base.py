"""
Model adapter interface — THE extension point for hooking any future model into the GUI.

The GUI's contract with a model is deliberately tiny:

    image file in  ->  DOTA-style .txt out
                       (x1 y1 x2 y2 x3 y3 x4 y4 class_name confidence, one box per line,
                        coordinates in the SAME pixel space as the input image)

Anything that can satisfy that — a subprocess, a REST call wrapped in a small script,
whatever — can be an adapter. The GUI runs adapters as subprocesses through a serial job
queue (see app/jobs.py), streams their logs to the browser, and picks up the prediction
file when they exit 0. See docs/ADAPTERS.md for a step-by-step guide with a template.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ModelAdapter(ABC):
    #: unique id, used in the API (letters/digits/_ only)
    name: str = ""
    #: human-readable name shown in the UI dropdown
    display_name: str = ""
    #: shown in the UI — say what the model does and what input it expects
    #: (e.g. required resolution / pre-processing), because the GUI can't check that.
    description: str = ""
    #: class names the model can emit (informational; used for UI color legend ordering)
    classes: list[str] = []
    #: options rendered in the "Run inference" dialog and passed to build_command as
    #: `extra`. Each: {"name", "label", "type": "bool"|"str", "default", "help"}
    options: list[dict] = []

    @abstractmethod
    def build_command(self, image_path: Path, out_dir: Path, work_dir: Path,
                      device: str, extra: dict) -> list[str]:
        """argv to run. Must write `<image stem>.txt` (DOTA format) into out_dir.

        out_dir / work_dir are job-private empty directories (safe for scratch); the
        adapter must not depend on global mutable state so jobs can't corrupt each other.
        """

    def pred_file_for(self, image_path: Path, out_dir: Path) -> Path:
        """Where the prediction lands after a successful run (override if different)."""
        return out_dir / f"{image_path.stem}.txt"

    def to_json(self) -> dict:
        return dict(name=self.name, display_name=self.display_name,
                    description=self.description, classes=self.classes,
                    options=self.options)
