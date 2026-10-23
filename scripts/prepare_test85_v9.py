#!/usr/bin/env python3
"""Assemble the October Test85 VisionBoard runtime tree without copying scene data."""
from __future__ import annotations

import re
from pathlib import Path


AUTORA = Path(__file__).resolve().parents[2]
DATASET = AUTORA / "DATA/LR_Data_06oct26"
DEPLOY = AUTORA / "deploy_server/deploy05oct26"
NEW6 = AUTORA / "deploy_server/run_new6_15oct26/lowres"
OUTPUT = DATASET / "VisionBoard_Test85_Oct2026"


def scenes() -> list[tuple[str, str, str]]:
    pattern = re.compile(r"^\|\s*([0-9A-Z]{2})\s*\|\s*`([^`]+)`\s*\|\s*(test[12])\s*\|")
    result = []
    for line in (DATASET / "Test85.md").read_text().splitlines():
        match = pattern.match(line)
        if match:
            result.append((match.group(1), match.group(2), match.group(3)))
    if len(result) != 85:
        raise RuntimeError(f"expected 85 scenes, found {len(result)}")
    return result


def first_file(relative: str) -> Path:
    for root in (DEPLOY, NEW6):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(relative)


def link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace {destination}")
    destination.symlink_to(source.resolve())


def main() -> None:
    for folder in (
        "images", "predictions/stage1", "predictions/stage2", "gt/labels_simple",
        "gt/labels_detailed", "gt/labels_delete", "heatmaps/stage0", "classes",
    ):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

    link(DEPLOY / "classes/stage1.txt", OUTPUT / "classes/stage1.txt")
    link(DEPLOY / "classes/stage2.txt", OUTPUT / "classes/stage2.txt")

    for _image_id, stem, split in scenes():
        link(DATASET / split / "images2x" / f"{stem}.png", OUTPUT / "images" / f"{stem}.png")
        link(DATASET / split / "labels_simple" / f"{stem}.txt",
             OUTPUT / "gt/labels_simple" / f"{stem}.txt")
        link(DATASET / split / "labels_detailed" / f"{stem}.txt",
             OUTPUT / "gt/labels_detailed" / f"{stem}.txt")
        link(DATASET / split / "labels_delete" / f"{stem}.txt",
             OUTPUT / "gt/labels_delete" / f"{stem}.txt")
        link(first_file(f"predictions/stage1/{stem}.txt"),
             OUTPUT / "predictions/stage1" / f"{stem}.txt")
        link(first_file(f"predictions/stage2/{stem}.txt"),
             OUTPUT / "predictions/stage2" / f"{stem}.txt")
        link(first_file(f"heatmaps/stage0/{stem}.npy"),
             OUTPUT / "heatmaps/stage0" / f"{stem}.npy")

    print(f"ready: {OUTPUT}")
    for folder in ("images", "predictions/stage1", "predictions/stage2",
                   "gt/labels_simple", "gt/labels_detailed", "gt/labels_delete",
                   "heatmaps/stage0"):
        print(f"{folder}: {len(list((OUTPUT / folder).iterdir()))}")


if __name__ == "__main__":
    main()
