"""Building, running and saving training jobs (in parallel, resumable)."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from . import config as C


def make_cfg(experiment, method, seed, shots=None):
    """experiment: 'noiseless', 'shots1000' or 'shots500'."""
    cfg = dict(**C.TASK, **C.DATA, **C.TRAIN, method=method, seed=seed,
               mode="shots" if shots else "analytic", shots=shots,
               loss_threshold=C.LOSS_THRESHOLD, acc_target=C.ACC_TARGET)
    cfg["digits"] = list(cfg["digits"])
    return experiment, cfg


def run_path(experiment, cfg) -> Path:
    return C.RESULTS / "runs" / experiment / cfg["method"] / f"seed_{cfg['seed']:03d}.json"


def is_done(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("complete"))
    except Exception:
        return False


def execute(task):
    """Worker: run one job and save it (write to .tmp, then rename = never half-written)."""
    experiment, cfg = task
    import torch
    torch.set_num_threads(1)
    from .train import train_run

    path = run_path(experiment, cfg)
    if is_done(path):
        return f"skip {path.relative_to(C.ROOT)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        rec = train_run(cfg)
        rec["experiment"] = experiment
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, path)
        return f"done {path.relative_to(C.ROOT)} {time.time() - t:.0f}s"
    except Exception:
        path.with_suffix(".error.txt").write_text(traceback.format_exc())
        return f"FAIL {path.relative_to(C.ROOT)}"


def run_tasks(tasks, workers, log):
    """Run jobs on `workers` CPU cores; finished jobs are skipped (resumable)."""
    from multiprocessing import Pool

    todo = [t for t in tasks if not is_done(run_path(*t))]
    log(f"{len(tasks)} jobs, {len(tasks) - len(todo)} already done, {len(todo)} to run")
    fails, t0 = 0, time.time()
    if todo:
        with Pool(workers, maxtasksperchild=20) as pool:
            for i, msg in enumerate(pool.imap_unordered(execute, todo), 1):
                fails += msg.startswith("FAIL")
                el = time.time() - t0
                log(f"[{i}/{len(todo)}] {msg} | {el / 60:.1f} min, eta {el / i * (len(todo) - i) / 60:.1f} min")
    return fails
