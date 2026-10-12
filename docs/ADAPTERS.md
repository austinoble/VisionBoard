# Connecting a model (adapters)

Every model you train plugs into the GUI through one small adapter file. The GUI calls it
as a subprocess, shows its log live, and picks up its output file — that's the whole
integration surface.

## The contract

> Given an image path, write **`<image-stem>.txt`** into the provided output directory,
> DOTA format, one detection per line:
>
> ```
> x1 y1 x2 y2 x3 y3 x4 y4 class_name confidence
> ```
>
> Coordinates in the **same pixel space as the input image** (the GUI overlays them 1:1).
> Axis-aligned boxes are fine — emit the 4 rectangle corners. Empty file = no detections.
> Exit code 0 on success, non-zero on failure.

If your model already writes DOTA files, the adapter is
pure glue. If it writes something else (COCO json, YOLO txt, …), convert inside a small
wrapper script and point the adapter at that.

## Step by step

**1. Create `app/adapters/my_model.py`:**

```python
from pathlib import Path
from .base import ModelAdapter

class MyModelAdapter(ModelAdapter):
    name = "my_model"                       # unique id (api-safe)
    display_name = "MyModel v3 (fold 0)"    # UI dropdown text
    description = (
        "One paragraph shown in the Run-inference dialog. State what input the "
        "model expects (resolution, SR level, bands) — the GUI cannot check this."
    )
    classes = ["tank", "vehicle"]           # informational
    options = [                             # rendered as form fields, passed via `extra`
        dict(name="fast", label="Fast mode (fold 0 only)", type="bool", default=False,
             help="Trades recall for ~5x speed."),
    ]

    def build_command(self, image_path: Path, out_dir: Path, work_dir: Path,
                      device: str, extra: dict) -> list[str]:
        cmd = ["/opt/my_env/bin/python", "/opt/my_model/predict.py",
               "--image", str(image_path),
               "--out-dir", str(out_dir),      # write <stem>.txt here
               "--tmp", str(work_dir),         # job-private scratch, auto-cleaned
               "--device", device]
        if extra.get("fast"):
            cmd.append("--fast")
        return cmd
```

**2. Register it in `app/adapters/__init__.py`:**

```python
from .my_model import MyModelAdapter
_all = [
    DinoEnsembleAdapter(),
    MyModelAdapter(),
]
```

**3. Restart the GUI.** The model appears in the Run-inference dropdown. Done.

## Rules the GUI relies on

- **Only touch `out_dir` / `work_dir`.** They are private to the job (`data/jobs/<id>/`),
  so parallel-submitted jobs (queued serially) can never corrupt each other. Don't write
  into your model's repo from the adapter.
- **Print progress to stdout/stderr** — it streams into the Jobs panel.
- If your output lands somewhere else or under another name, override
  `pred_file_for(image_path, out_dir)` to return the actual path; the GUI copies it to the
  scene's canonical prediction slot afterwards.
- Long environment activation, multi-env orchestration, etc. belong inside your own driver
  script (see how `run_pipeline.py` shells between two conda envs) — the adapter should
  stay a one-command wrapper.

## Testing an adapter without a GPU

```python
import sys; sys.path.insert(0, "/absolute/path/to/VisionBoard")
from app import jobs
j = jobs.submit("uploads", "<some-uploaded-stem>", "my_model", device="0", extra={})
# poll jobs.get(j["id"]) until state == "done"; log in jobs.log_tail(j["id"])
```

Or use the GUI itself — upload a small crop and run the model on it.
