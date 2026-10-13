"""All experiment settings (fixed before the comparison runs)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# The task: real MNIST digits 4 vs 9, 10 qubits, 16 layers, readout <Z_0> (local cost)
TASK = dict(n_qubits=10, n_layers=16, digits=(4, 9), cost="local")
DATA = dict(n_train=600, n_val=200, n_test=400)

TRAIN = dict(
    batch_size=16,          # images per update
    updates_per_stage=20,   # LL adds a layer every 20 updates -> 16 x 20 = 320 updates
    lr=0.03,                # Adam learning rate (chosen on validation data in a pilot sweep: 0.01 / 0.03 / 0.1)
    init_range=3.141592653589793,   # Direct: every angle ~ U(-pi, pi)
    ident_range=0.1,        # LL: added layers start near identity, angles ~ U(-0.1, 0.1)
)

METHODS = ("direct", "ll")  # ll = layer-wise learning (all present layers keep training)
SEEDS = tuple(range(30))
SHOTS_TRAIN = (500, 1000)   # finite-shot (sampling-noise) training

LOSS_THRESHOLD = 0.3466     # ln(2)/2: half the loss of a coin-flip classifier
ACC_TARGET = 0.90           # success = final train loss < threshold AND test accuracy >= 0.90
