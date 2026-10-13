"""Direct training and layer-wise training (LL) of the quantum classifier.

Methods (L = 16 layers, s = 20 updates per stage, T = L*s = 320 updates)
  direct  all L layers present and trainable from the start, angles ~ U(-pi, pi);
          T updates.
  ll      start with 1 layer (identical to Direct's first layer for the same seed),
          train s updates, append a near-identity layer (angles ~ U(-0.1, 0.1)),
          train s updates, ... until L layers. All present layers keep training.
          Same total budget T as Direct.

Gradient modes
  analytic  exact gradients (noiseless simulation)
  shots     parameter-shift gradients in which every circuit expectation value is
            estimated from a finite number of measurement shots (sampling noise)

The classical head p = sigmoid(a * <Z0> + b) is always trainable.
"""
from __future__ import annotations

import math
import platform
import time
from datetime import datetime, timezone

import numpy as np
import torch

from .data import make_split
from .fastsim import Engine, sample_expval

EPS = 1e-12


def versions():
    import pennylane, sklearn, scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "pennylane": pennylane.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}


def is_ll(method):
    return method in ("ll", "ll_joint")       # "ll_joint" = name used in the original run files


def schedule(method, L, s):
    """List of training stages (depth, trainable_layers, n_updates)."""
    if method == "direct":
        return [(L, list(range(L)), L * s)]
    if is_ll(method):
        return [(d, list(range(d)), s) for d in range(1, L + 1)]
    raise ValueError(method)


def init_weights(method, L, n, seed, init_range=math.pi, ident_range=0.1):
    """Direct: every layer U(-pi, pi). LL: layer 0 identical to Direct's layer 0
    for the same seed (paired comparison); later layers U(-0.1, 0.1)."""
    w_rand = np.random.default_rng([seed, 1]).uniform(-init_range, init_range, (L, n, 2))
    if method == "direct":
        return w_rand
    w = np.random.default_rng([seed, 2]).uniform(-ident_range, ident_range, (L, n, 2))
    w[0] = w_rand[0]
    return w


def bce(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MaskedAdam:
    """Adam in which only the currently trainable entries are updated.

    Each entry keeps its own step counter, so a layer that starts training late
    gets fresh, correctly bias-corrected moments.
    """

    def __init__(self, shapes, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = [np.zeros(s) for s in shapes]

    def step(self, params, grads, masks):
        for i, (p, g, mk) in enumerate(zip(params, grads, masks)):
            mk = np.asarray(mk, dtype=bool)
            self.t[i][mk] += 1
            self.m[i][mk] = self.b1 * self.m[i][mk] + (1 - self.b1) * g[mk]
            self.v[i][mk] = self.b2 * self.v[i][mk] + (1 - self.b2) * g[mk] ** 2
            mh = self.m[i][mk] / (1 - self.b1 ** self.t[i][mk])
            vh = self.v[i][mk] / (1 - self.b2 ** self.t[i][mk])
            p[mk] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class Model:
    def __init__(self, n_qubits, n_layers, cost, weights):
        self.eng = Engine(n_qubits, cost)
        self.w = np.array(weights, dtype=np.float64)
        self.a = np.array([1.0])
        self.b = np.array([0.0])

    def expval(self, x, depth):
        with torch.no_grad():
            return self.eng.expval_chunked(x, torch.as_tensor(self.w[:depth])).numpy()

    def metrics(self, y, f):
        p = sigmoid(self.a[0] * f + self.b[0])
        return bce(p, y), float(np.mean((p >= 0.5) == (y >= 0.5)))

    def grads_analytic(self, x, y, depth):
        """Exact gradient of the minibatch loss (automatic differentiation)."""
        w = torch.tensor(self.w[:depth], requires_grad=True)
        a = torch.tensor(self.a[0], requires_grad=True)
        b = torch.tensor(self.b[0], requires_grad=True)
        z = a * self.eng.expval(torch.as_tensor(x), w) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.as_tensor(y))
        gw, ga, gb = torch.autograd.grad(loss, (w, a, b))
        g = np.zeros_like(self.w)
        g[:depth] = gw.numpy()
        return loss.item(), g, np.array([ga.item()]), np.array([gb.item()])

    def grads_shots(self, x, y, depth, trainable, shots, rng):
        """Parameter-shift gradient with every expectation estimated from `shots` shots.

        The forward value f_i and each shifted value f_i(theta +- pi/2 e_k) are
        independent S-shot estimates; the classical chain rule then uses them
        exactly as a shot-based quantum computer + classical optimiser would.
        """
        w = torch.as_tensor(self.w[:depth])
        f0, fp, fm = self.eng.shift_values(torch.as_tensor(x), w, trainable)
        f_hat = sample_expval(f0.numpy(), shots, rng)
        fp_hat = sample_expval(fp.numpy(), shots, rng)
        fm_hat = sample_expval(fm.numpy(), shots, rng)
        df = 0.5 * (fp_hat - fm_hat)                          # parameter-shift rule
        p = sigmoid(self.a[0] * f_hat + self.b[0])
        loss = bce(p, y)
        dz = (p - y) / len(y)                                 # d loss / d z
        g = np.zeros_like(self.w)
        g[trainable] = np.einsum("i,ijkl->jkl", dz * self.a[0], df)
        return loss, g, np.array([np.sum(dz * f_hat)]), np.array([np.sum(dz)])


def circuit_evals(n_trainable, batch, mode):
    """Number of circuit executions needed for one gradient estimate."""
    if mode == "shots":
        return int(batch * (1 + 2 * int(n_trainable)))
    return int(batch)


def train_run(cfg: dict) -> dict:
    """Run one training job described by cfg; return the full result record."""
    t_start = time.time()
    n, L = cfg["n_qubits"], cfg["n_layers"]
    split = make_split(tuple(cfg["digits"]), n_features=n, n_train=cfg["n_train"],
                       n_val=cfg["n_val"], n_test=cfg["n_test"])
    stages = schedule(cfg["method"], L, cfg["updates_per_stage"])
    model = Model(n, L, cfg["cost"],
                  init_weights(cfg["method"], L, n, cfg["seed"], cfg["init_range"], cfg["ident_range"]))
    opt = MaskedAdam([model.w.shape, (1,), (1,)], cfg["lr"])
    mode = cfg.get("mode", "analytic")
    batch_rng = np.random.default_rng([cfg["seed"], 3])
    shot_rng = np.random.default_rng([cfg["seed"], 4, int(cfg.get("shots") or 0)])
    B = cfg["batch_size"]
    xtr, ytr = split.x_train, split.y_train
    order, ptr = batch_rng.permutation(len(ytr)), 0

    H = {k: [] for k in ("iter", "depth", "train_loss", "train_acc", "val_loss", "val_acc",
                         "test_loss", "test_acc", "batch_loss", "grad_norm_trainable",
                         "grad_norm_active", "grad_mean_abs_trainable", "circuit_evals")}
    stage_starts = []
    x_all = np.concatenate([xtr, split.x_val, split.x_test])
    n1, n2 = len(ytr), len(ytr) + len(split.y_val)

    def record(it, depth, batch_loss, gn_tr, gn_act, gmean, cev):
        f = model.expval(x_all, depth)          # evaluate train + validation + test
        trl, tra = model.metrics(ytr, f[:n1])
        vl, va = model.metrics(split.y_val, f[n1:n2])
        tel, tea = model.metrics(split.y_test, f[n2:])
        for k, v in zip(H.keys(), (it, depth, trl, tra, vl, va, tel, tea, batch_loss,
                                   gn_tr, gn_act, gmean, cev)):
            H[k].append(v)

    record(0, stages[0][0], None, None, None, None, 0)
    it, total_cev = 0, 0
    for depth, trainable, n_up in stages:
        stage_starts.append({"iter": it, "depth": depth, "trainable_layers": trainable,
                             "updates": n_up})
        mask_w = np.zeros(model.w.shape, dtype=bool)
        mask_w[trainable] = True
        for _ in range(n_up):
            if ptr + B > len(ytr):
                order, ptr = batch_rng.permutation(len(ytr)), 0
            idx = order[ptr:ptr + B]
            ptr += B
            if mode == "shots":
                loss, g, ga, gb = model.grads_shots(xtr[idx], ytr[idx], depth, trainable,
                                                    cfg["shots"], shot_rng)
                gn_act = None
            else:
                loss, g, ga, gb = model.grads_analytic(xtr[idx], ytr[idx], depth)
                gn_act = float(np.linalg.norm(g[:depth]))
            gtr = g[mask_w]
            total_cev += circuit_evals(mask_w.sum(), B, mode)
            opt.step([model.w, model.a, model.b], [g, ga, gb],
                     [mask_w, np.ones(1, bool), np.ones(1, bool)])
            it += 1
            record(it, depth, loss, float(np.linalg.norm(gtr)), gn_act,
                   float(np.mean(np.abs(gtr))), total_cev)

    thr = cfg["loss_threshold"]
    final = {k: H[k][-1] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                                   "test_loss", "test_acc")}
    summary = {
        **{"final_" + k: v for k, v in final.items()},
        "success": bool(final["train_loss"] < thr and final["test_acc"] >= cfg["acc_target"]),
        "iter_converge_loss": next((i for i, l in zip(H["iter"], H["train_loss"]) if l < thr), None),
        "iter_reach_test_acc": next((i for i, a in zip(H["iter"], H["test_acc"])
                                     if a >= cfg["acc_target"]), None),
        "total_updates": it,
        "final_depth_updates": it - stage_starts[-1]["iter"],
        "total_circuit_evals": int(total_cev),
        "wall_time_s": time.time() - t_start,
    }
    return {
        "config": cfg,
        "data_info": split.info,
        "versions": versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stages": stage_starts,
        "history": H,
        "summary": summary,
        "final_params": {"weights": model.w.tolist(), "scale": float(model.a[0]),
                         "bias": float(model.b[0])},
        "complete": True,
    }
