"""Run the experiments and make the figures.

Usage (from the project folder, with the venv's python):
    python run_all.py figures      # summary table + all figures from the saved results (~1 min)
    python run_all.py train        # Direct and LL, 30 seeds: noiseless, 1000 shots, 500 shots (~5 h)
    python run_all.py diagnostic   # barren-plateau gradient-variance diagnostic (~1 min)
Training skips every run whose result file already exists, so with the saved
results in results/ only `figures` does any work.
Options: --workers N (default: number of CPU cores)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def train_tasks():
    from src.runner import make_cfg
    tasks = []
    for experiment, shots in (("noiseless", None), ("shots1000", 1000), ("shots500", 500)):
        for m in C.METHODS:
            for s in C.SEEDS:
                tasks.append(make_cfg(experiment, m, s, shots=shots))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "diagnostic", "figures"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.stage == "train":
        from src.runner import run_tasks
        sys.exit(1 if run_tasks(train_tasks(), a.workers, log) else 0)
    if a.stage == "diagnostic":
        from src.gradients import run_diagnostic
        run_diagnostic(log, a.workers)
    if a.stage == "figures":
        from src.figures import make_all
        make_all(log)


if __name__ == "__main__":
    main()
