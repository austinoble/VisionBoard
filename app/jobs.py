"""
Inference job queue: runs model adapters as subprocesses, one at a time.

Serial on purpose — the models here saturate a GPU by themselves, and the wrapped engines
may assume exclusive use of their checkpoints/VRAM. Each job gets a private directory
under data/jobs/<job_id>/ containing:

    job.json    spec + state (survives server restarts, so history persists)
    log.txt     combined stdout/stderr of the subprocess, streamed to the UI
    out/        adapter writes the prediction file here
    work/       adapter scratch (tiles, intermediate pickles, ...)

On success the prediction file is copied to the scene's canonical location (see
store.generated_pred_target) so it immediately shows up as an overlay, and out/ + work/
are cleaned up.

States: queued -> running -> done | failed | error(couldn't launch)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import config, store
from .adapters import ADAPTERS

_queue: list[str] = []
_lock = threading.Lock()
_worker_started = False


def _job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


def _read(job_id: str) -> dict | None:
    p = _job_dir(job_id) / "job.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _write(job: dict) -> None:
    d = _job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(job, indent=1))


def submit(source: str, stem: str, adapter_name: str, device: str, extra: dict) -> dict:
    """Validate and enqueue an inference job. Returns the job record."""
    if adapter_name not in ADAPTERS:
        raise ValueError(f"unknown adapter '{adapter_name}'")
    scene = store.get_scene(source, stem)
    if not scene:
        raise ValueError(f"unknown scene {source}/{stem}")

    job = dict(
        id=uuid.uuid4().hex[:12],
        source=source, stem=stem,
        adapter=adapter_name, device=device, extra=extra,
        state="queued", created=time.time(), started=None, finished=None,
        error=None, pred_target=str(store.generated_pred_target(store.get_source(source), stem)),
    )
    _write(job)
    with _lock:
        _queue.append(job["id"])
    _ensure_worker()
    return job


def get(job_id: str) -> dict | None:
    return _read(job_id)


def log_tail(job_id: str, max_bytes: int = 16384) -> str:
    p = _job_dir(job_id) / "log.txt"
    if not p.is_file():
        return ""
    size = p.stat().st_size
    with open(p, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode(errors="replace")


def list_jobs(limit: int = 50) -> list[dict]:
    if not config.JOBS_DIR.is_dir():
        return []
    jobs = [j for d in config.JOBS_DIR.iterdir() if (j := _read(d.name))]
    jobs.sort(key=lambda j: -j["created"])
    return jobs[:limit]


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_worker, daemon=True).start()


def _worker() -> None:
    while True:
        with _lock:
            job_id = _queue.pop(0) if _queue else None
        if job_id is None:
            time.sleep(0.5)
            continue
        try:
            _run(job_id)
        except Exception as e:  # never let the worker die — later jobs must still run
            job = _read(job_id)
            if job:
                job.update(state="error", error=str(e), finished=time.time())
                _write(job)


def _run(job_id: str) -> None:
    job = _read(job_id)
    adapter = ADAPTERS[job["adapter"]]
    scene = store.get_scene(job["source"], job["stem"])
    if not scene:
        raise RuntimeError("scene disappeared")

    d = _job_dir(job_id)
    out_dir, work_dir = d / "out", d / "work"
    out_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    cmd = adapter.build_command(scene["image_path"], out_dir, work_dir,
                                job["device"], job.get("extra") or {})
    job.update(state="running", started=time.time(), command=" ".join(cmd))
    _write(job)

    with open(d / "log.txt", "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(d))
        rc = proc.wait()

    if rc != 0:
        job.update(state="failed", error=f"exit code {rc}", finished=time.time())
        _write(job)
        return

    pred = adapter.pred_file_for(scene["image_path"], out_dir)
    if not pred.is_file():
        job.update(state="failed", finished=time.time(),
                   error=f"adapter exited 0 but prediction file missing: {pred.name}")
        _write(job)
        return

    target = Path(job["pred_target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pred, target)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    job.update(state="done", finished=time.time())
    _write(job)
