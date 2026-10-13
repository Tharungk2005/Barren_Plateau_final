"""Barren-plateau diagnostic: gradient variance over random initialisations.

Cost per circuit: C(theta) = mean over a fixed batch of 16 real MNIST training
images (digits of the hard config) of <O>, with O = Z_0 (local) or
Z_0 Z_1 ... Z_{n-1} (global). For each (n_qubits, depth, cost) we draw
N_INIT = 100 parameter vectors theta ~ U(-pi, pi) and compute the exact
gradient. Reported:
  var_fixed  Var_theta[dC/dtheta_(layer 0, qubit 0, RY)]  (fixed component)
  var_mean   mean over all components of Var_theta[dC/dtheta_k]
  95% bootstrap confidence intervals (2000 resamples of the 100 inits).
Additionally (local cost, depth sweep) the same quantities for the layer-wise
initialisation (layer 0 ~ U(-pi, pi), later layers ~ U(-0.1, 0.1)).
"""
from __future__ import annotations

import json
import math
from multiprocessing import Pool

import numpy as np

from . import config as C

N_INIT = 100
BATCH = 16
QUBITS = (4, 6, 8, 10, 12)
FIXED_DEPTH = C.TASK["n_layers"]
DEPTH_QUBITS = C.TASK["n_qubits"]
DEPTHS = tuple(range(1, 21))
OUT = C.RESULTS / "diagnostic"


def _point(args):
    sweep, n, L, cost, init = args
    import torch
    torch.set_num_threads(1)
    from .data import make_split
    from .fastsim import Engine

    sp = make_split(tuple(C.TASK["digits"]), n_features=n)
    x = torch.tensor(sp.x_train[:BATCH])
    eng = Engine(n, cost)
    grads = np.zeros((N_INIT, L, n, 2))
    for i in range(N_INIT):
        rng = np.random.default_rng([n, L, i, 0 if cost == "local" else 1, 0 if init == "random" else 1])
        if init == "random":
            w = rng.uniform(-math.pi, math.pi, (L, n, 2))
        else:
            w = rng.uniform(-0.1, 0.1, (L, n, 2))
            w[0] = rng.uniform(-math.pi, math.pi, (n, 2))
        wt = torch.tensor(w, requires_grad=True)
        grads[i] = torch.autograd.grad(eng.expval(x, wt).mean(), wt)[0].numpy()
    g_fixed = grads[:, 0, 0, 0]
    flat = grads.reshape(N_INIT, -1)
    boot = np.random.default_rng(12345)
    bi = boot.integers(0, N_INIT, (2000, N_INIT))
    bf = np.var(g_fixed[bi], axis=1, ddof=1)
    bm = np.array([np.var(flat[b], axis=0, ddof=1).mean() for b in bi[:500]])
    return {"sweep": sweep, "n_qubits": n, "depth": L, "cost": cost, "init": init,
            "n_init": N_INIT, "batch": BATCH,
            "var_fixed": float(np.var(g_fixed, ddof=1)),
            "var_fixed_ci95": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
            "var_mean": float(np.var(flat, axis=0, ddof=1).mean()),
            "var_mean_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "mean_abs_grad": float(np.abs(flat).mean()),
            "grad_fixed_samples": g_fixed.tolist()}


def tasks():
    t = []
    for cost in ("local", "global"):
        for n in QUBITS:
            t.append(("qubits", n, FIXED_DEPTH, cost, "random"))
        for L in DEPTHS:
            t.append(("depth", DEPTH_QUBITS, L, cost, "random"))
    for L in DEPTHS:
        t.append(("depth", DEPTH_QUBITS, L, "local", "layerwise"))
    return t


def run_diagnostic(log=print, workers=8):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bp_diagnostic.json"
    if path.exists():
        log("diagnostic already complete, skipping")
        return
    todo = sorted(tasks(), key=lambda a: -(2 ** a[1]) * a[2])
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_point, todo):
            rows.append(r)
            log(f"diag {r['sweep']} n={r['n_qubits']} L={r['depth']} {r['cost']} {r['init']}: "
                f"var_fixed={r['var_fixed']:.3e} var_mean={r['var_mean']:.3e}")
    rows.sort(key=lambda r: (r["sweep"], r["cost"], r["init"], r["n_qubits"], r["depth"]))
    from .train import versions
    path.write_text(json.dumps({"description": __doc__, "versions": versions(),
                                "data_digits": list(C.TASK["digits"]),
                                "points": rows}, indent=1))
    log(f"diagnostic written: {path}")
