"""Results table and figures for Direct vs Layer-wise (LL) training.

Reads only the saved run files in results/ (nothing is retrained) and writes
    results/summary.csv          one row per (noise setting, method)
    figures/fig1_circuit.png     the variational circuit
    figures/fig2_training_schemes.png   which layers are trained when
    figures/fig3_barren_plateau.png     gradient variance vs qubits / depth
    figures/fig4_test_accuracy_curves.png
    figures/fig5_final_accuracy.png
    figures/fig6_success_rate.png
    figures/fig7_iterations_to_converge.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from . import config as C  # noqa: E402

METHODS = {"direct": "Direct", "ll": "Layer-wise (LL)"}
COLOR = {"direct": "#eb6834", "ll": "#2a78d6"}
SETTINGS = {"noiseless": "Noiseless\n(exact gradients)",
            "shots1000": "1000 shots\n(sampling noise)",
            "shots500": "500 shots\n(more noise)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ------------------------------------------------------------------ data
def load_runs(setting, method):
    files = sorted((C.RESULTS / "runs" / setting / method).glob("seed_*.json"))
    return [json.loads(f.read_text()) for f in files]


def load_all():
    return {(s, m): load_runs(s, m) for s in SETTINGS for m in METHODS}


def load_runs_sgd(setting, method):
    folder = C.RESULTS / "runs_sgd" / setting / method
    if not folder.exists():
        return []
    files = sorted(folder.glob("seed_*.json"))
    return [json.loads(f.read_text()) for f in files]


def load_all_sgd():
    res = {}
    for s in SETTINGS:
        for m in METHODS:
            p = C.RESULTS / "runs_sgd" / s / m
            if p.exists():
                r = load_runs_sgd(s, m)
                if r:
                    res[(s, m)] = r
    return res


def ci95(a):
    a = np.asarray(a, float)
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def summarize(runs):
    rows = []
    for (s, m), R in runs.items():
        S = [r["summary"] for r in R]
        acc = np.array([x["final_test_acc"] for x in S])
        reach = [x["iter_reach_test_acc"] for x in S if x["iter_reach_test_acc"] is not None]
        rows.append({
            "setting": s, "method": m, "runs": len(R),
            "mean_test_acc": round(float(acc.mean()), 4), "ci95_test_acc": round(float(ci95(acc)), 4),
            "mean_final_train_loss": round(float(np.mean([x["final_train_loss"] for x in S])), 4),
            "successful_runs": int(sum(x["success"] for x in S)),
            "success_rate_pct": round(float(100 * np.mean([x["success"] for x in S])), 1),
            "runs_reaching_90pct": len(reach),
            "median_iters_to_90pct": float(np.median(reach)) if reach else None,
        })
    return rows


def write_summary(rows, path=None):
    path = path or C.RESULTS / "summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _row(rows, s, m):
    return next(r for r in rows if r["setting"] == s and r["method"] == m)


# --------------------------------------------------------------- circuits
def _gate(ax, x, y, text, fc, w=0.9, h=0.62, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=INK, lw=1, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def fig_circuit(path):
    n = C.TASK["n_qubits"]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_axis_off()
    ys = [n - 1 - q for q in range(n)]
    for q, y in enumerate(ys):
        ax.plot([0.6, 10.6 if q == 0 else 9.9], [y, y], color=INK, lw=1, zorder=1)
        ax.text(0.45, y, rf"$q_{{{q}}}$: $|0\rangle$", ha="right", va="center", fontsize=10)
        _gate(ax, 1.5, y, rf"$R_Y(x_{{{q}}})$", "#d9efe4", w=1.2)        # data encoding
        _gate(ax, 3.4, y, r"$R_Y(\theta)$", "#dce9f8")                  # trainable rotations
        _gate(ax, 4.6, y, r"$R_Z(\theta)$", "#dce9f8")
    # CZ ring: (0,1),(2,3).. | (1,2),(3,4).. | (9,0); CZ gates commute, so this order is exact
    cols = {5.8: [(q, q + 1) for q in range(0, n - 1, 2)],
            6.6: [(q, q + 1) for q in range(1, n - 1, 2)],
            7.4: [(n - 1, 0)]}
    for x, pairs in cols.items():
        for a, b in pairs:
            ax.plot([x, x], [ys[a], ys[b]], color=INK, lw=1.2, zorder=2)
            ax.plot([x, x], [ys[a], ys[b]], "o", color=INK, ms=6, zorder=3)
    # one-layer bracket
    ax.add_patch(Rectangle((2.75, -0.55), 5.1, n + 0.1, fill=False, ec=COLOR["ll"],
                           lw=1.8, ls="--", zorder=0))
    ax.text(5.3, n - 0.15, "one trainable layer\n(repeated L = 16 times)", ha="center",
            va="bottom", fontsize=11, color=COLOR["ll"], fontweight="bold")
    ax.text(1.5, n - 0.15, "data encoding", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(6.6, -0.95, "CZ ring (entangling)", ha="center", va="top", fontsize=10, color=MUTED)
    ax.text(4.0, -0.95, "20 angles per layer", ha="center", va="top", fontsize=10, color=MUTED)
    ax.text(9.1, -0.95, "... layers 2-16 ...", ha="center", va="top", fontsize=10, color=MUTED)
    for y in ys:
        ax.text(9.1, y, "...", ha="center", va="center", fontsize=12, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=1), zorder=3)
    # measurement on qubit 0 and classical head
    _gate(ax, 10.6, ys[0], r"measure $\langle Z_0\rangle$", "#fbe3d6", w=1.4)
    ax.text(10.6, n - 0.15, "readout", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(12.9, ys[0] - 1.5), xytext=(10.9, ys[0] - 0.35),
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(12.9, ys[0] - 1.6, r"$p(\mathrm{digit\ 9}\,|\,x) = \sigma(a\,\langle Z_0\rangle + b)$"
            "\nclassical head (a, b trained too)", ha="center", va="top", fontsize=10.5)
    ax.set_xlim(-0.6, 15.0)
    ax.set_ylim(-1.5, n + 1.2)
    ax.set_title("Quantum classifier: 10 qubits, 16 layers, MNIST digits 4 vs 9 "
                 "(10 PCA features, one per qubit)", pad=8)
    fig.savefig(path)
    plt.close(fig)


def fig_training_schemes(path):
    L, s = C.TASK["n_layers"], C.TRAIN["updates_per_stage"]
    T = L * s
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, m in zip(axes, METHODS):
        for l in range(1, L + 1):
            start = 0 if m == "direct" else (l - 1) * s
            new = s if (m == "ll" and l > 1) else 0
            ax.barh(l, T - start - new, left=start + new, height=0.78, color=COLOR[m])
            if new:
                ax.barh(l, new, left=start, height=0.78, color="#a9c9f0")
        ax.set_xlim(0, T)
        ax.set_ylim(0.4, L + 0.6)
        ax.set_xticks(range(0, T + 1, 40))
        ax.set_yticks([1, 4, 8, 12, 16])
        ax.set_xlabel("training update (Adam step, batch of 16 images)")
        ax.grid(axis="y", visible=False)
    axes[0].set_ylabel("circuit layer")
    axes[0].set_title("(a) Direct: all 16 layers trained from step 0\nrandom angles in [-π, π]")
    axes[1].set_title("(b) Layer-wise: start with 1 layer, add 1 layer every 20 updates\n"
                      "new layer starts near identity (angles in [-0.1, 0.1])")
    axes[1].text(6, L - 1.5, "light blue = layer just added (starts ≈ identity)\n"
                 "layers already present keep training", ha="left", va="center", fontsize=9.5,
                 color=MUTED)
    fig.suptitle("Both protocols get the same budget: 320 updates", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------- barren plateau
def fig_barren_plateau(path):
    pts = json.loads((C.RESULTS / "diagnostic" / "bp_diagnostic.json").read_text())["points"]

    def series(**kw):
        P = sorted([p for p in pts if all(p[k] == v for k, v in kw.items())],
                   key=lambda p: (p["n_qubits"], p["depth"]))
        return P

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    P = series(sweep="qubits", cost="global", init="random")
    x = [p["n_qubits"] for p in P]
    ax.plot(x, [p["var_fixed"] for p in P], "-o", color=COLOR["direct"], lw=2, ms=7)
    ax.fill_between(x, [p["var_fixed_ci95"][0] for p in P], [p["var_fixed_ci95"][1] for p in P],
                    color=COLOR["direct"], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xlabel("number of qubits")
    ax.set_ylabel(r"Var[$\partial C / \partial\theta$]  (log scale)")
    ax.set_title("(a) Barren plateau: gradients vanish\nexponentially with qubit count "
                 "(16 layers, random init)")

    ax = axes[1]
    for init, m, lab in (("random", "direct", "random init (Direct)"),
                         ("layerwise", "ll", "layer-wise init (LL)")):
        P = series(sweep="depth", cost="local", init=init)
        x = [p["depth"] for p in P]
        ax.plot(x, [p["var_fixed"] for p in P], "-o", color=COLOR[m], lw=2, ms=5, label=lab)
        ax.fill_between(x, [p["var_fixed_ci95"][0] for p in P], [p["var_fixed_ci95"][1] for p in P],
                        color=COLOR[m], alpha=0.15, lw=0)
    r = {i: next(p for p in series(sweep="depth", cost="local", init=i) if p["depth"] == 16)["var_fixed"]
         for i in ("random", "layerwise")}
    ax.axvline(16, color=MUTED, lw=1, ls=":")
    ax.text(15.6, r["random"] * 1.6, f"depth 16:\n{r['layerwise'] / r['random']:.0f}x larger\nwith LL init",
            ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 8, 12, 16, 20])
    ax.set_xlabel("number of layers (10 qubits)")
    ax.set_title("(b) Our model: gradient variance vs depth\n"
                 "random init decays, LL init stays large")
    ax.legend(loc="lower left")
    fig.text(0.5, -0.04, "Gradient of the first rotation angle, 100 random initialisations on real MNIST "
             "inputs, shaded = 95% CI.  (a) global cost  <Z0 Z1 ... Zn-1>,  (b) local cost <Z0> "
             "(the cost used for training).", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------ performance
def fig_curves(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, s in zip(axes, SETTINGS):
        for m in METHODS:
            A = np.array([r["history"]["test_acc"] for r in runs[(s, m)]])
            it = np.array(runs[(s, m)][0]["history"]["iter"])
            mu, ci = A.mean(0), 1.96 * A.std(0, ddof=1) / np.sqrt(len(A))
            ax.plot(it, mu, color=COLOR[m], lw=2, label=METHODS[m])
            ax.fill_between(it, mu - ci, mu + ci, color=COLOR[m], alpha=0.18, lw=0)
        ax.axhline(C.ACC_TARGET, color=MUTED, lw=1, ls="--")
        ax.text(4, C.ACC_TARGET + 0.008, "90% target", ha="left", va="bottom", fontsize=9, color=MUTED)
        ax.set_title(SETTINGS[s].replace("\n", " "))
        ax.set_xlabel("training update")
        ax.set_xlim(0, 320)
        ax.set_xticks(range(0, 321, 80))
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].legend(loc="lower right")
    fig.suptitle("Test accuracy during training (mean of 30 seeds, shaded = 95% CI)", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _grouped_bars(ax, rows, key, err=None, fmt="{:.2f}", extra=None):
    x = np.arange(len(SETTINGS))
    w = 0.36
    for i, m in enumerate(METHODS):
        vals = [_row(rows, s, m)[key] or 0 for s in SETTINGS]
        errs = [_row(rows, s, m)[err] for s in SETTINGS] if err else None
        xs = x + (i - 0.5) * w
        ax.bar(xs, vals, w * 0.92, color=COLOR[m], label=METHODS[m], yerr=errs,
               error_kw=dict(ecolor=INK, capsize=4, lw=1))
        for j, (xx, v) in enumerate(zip(xs, vals)):
            top = v + (errs[j] if errs else 0)
            lab = fmt.format(v) + (("\n" + extra(_row(rows, list(SETTINGS)[j], m))) if extra else "")
            ax.text(xx, top, lab, ha="center", va="bottom", fontsize=9, color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(SETTINGS.values())
    ax.grid(axis="x", visible=False)


def fig_final_accuracy(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _grouped_bars(ax, rows, "mean_test_acc", err="ci95_test_acc")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("final test accuracy")
    ax.set_title("Final test accuracy after 320 updates\n(mean of 30 seeds, error bar = 95% CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_success(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _grouped_bars(ax, rows, "success_rate_pct", fmt="{:.0f}%",
                  extra=lambda r: f"({r['successful_runs']}/{r['runs']})")
    ax.set_ylim(0, 110)
    ax.set_ylabel("successful runs (%)")
    ax.set_title("Percentage of successful runs (30 seeds each)\n"
                 "success = final train loss < 0.347 and test accuracy ≥ 90%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_iterations(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _grouped_bars(ax, rows, "median_iters_to_90pct", fmt="{:.0f}",
                  extra=lambda r: f"reached in\n{r['runs_reaching_90pct']}/{r['runs']} runs")
    ax.set_ylim(0, 420)
    ax.axhline(320, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(-0.45, 324, "training budget = 320", ha="left", va="bottom", fontsize=9, color=MUTED)
    ax.set_ylabel("training updates (median)")
    ax.set_title("Iterations to converge: updates until test accuracy first reaches 90%\n"
                 "(median over the runs that reached it; lower is faster)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_curves_sgd(runs, path):
    settings_present = [s for s in SETTINGS if any((s, m) in runs and len(runs[(s, m)]) > 0 for m in METHODS)]
    if not settings_present:
        return
    fig, axes = plt.subplots(1, len(settings_present), figsize=(5.5 * len(settings_present), 4.4), sharey=True)
    if len(settings_present) == 1:
        axes = [axes]
    for ax, s in zip(axes, settings_present):
        for m in METHODS:
            if (s, m) not in runs or len(runs[(s, m)]) == 0:
                continue
            A = np.array([r["history"]["test_acc"] for r in runs[(s, m)]])
            it = np.array(runs[(s, m)][0]["history"]["iter"])
            mu = A.mean(0)
            ci = 1.96 * A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros_like(mu)
            ax.plot(it, mu, color=COLOR[m], lw=2, label=METHODS[m])
            if len(A) > 1 and ci.max() > 0:
                ax.fill_between(it, mu - ci, mu + ci, color=COLOR[m], alpha=0.18, lw=0)
        ax.axhline(C.ACC_TARGET, color=MUTED, lw=1, ls="--")
        ax.text(4, C.ACC_TARGET + 0.008, "90% target", ha="left", va="bottom", fontsize=9, color=MUTED)
        ax.set_title(f"{SETTINGS[s].replace(chr(10), ' ')}\n(Vanilla Gradient Descent)")
        ax.set_xlabel("training update")
        ax.set_xlim(0, 320)
        ax.set_xticks(range(0, 321, 80))
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0.35, 1.0)
    axes[0].legend(loc="lower right")
    fig.suptitle("Vanilla SGD: Test accuracy during training (shaded = 95% CI)", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_final_accuracy_sgd(rows, path):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    settings_present = list(dict.fromkeys(r["setting"] for r in rows))
    w = 0.36
    for i, m in enumerate(METHODS):
        vals = [next((r["mean_test_acc"] for r in rows if r["setting"] == s and r["method"] == m), 0) for s in settings_present]
        errs = [next((r["ci95_test_acc"] for r in rows if r["setting"] == s and r["method"] == m), 0) for s in settings_present]
        xs = np.arange(len(settings_present)) + (i - 0.5) * w
        ax.bar(xs, vals, w * 0.92, color=COLOR[m], label=METHODS[m], yerr=errs,
               error_kw=dict(ecolor=INK, capsize=4, lw=1))
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(settings_present)))
    ax.set_xticklabels([SETTINGS[s].replace("\n", " ") for s in settings_present])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("final test accuracy")
    ax.set_title("Vanilla SGD: Final test accuracy after 320 updates")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_comparison_adam_vs_sgd(runs_adam, runs_sgd, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    s = "noiseless"
    it = np.array(runs_adam[(s, "direct")][0]["history"]["iter"])

    # Panel 1: Adam
    ax1 = axes[0]
    for m in METHODS:
        if (s, m) in runs_adam:
            A = np.array([r["history"]["test_acc"] for r in runs_adam[(s, m)]])
            mu, ci = A.mean(0), 1.96 * A.std(0, ddof=1) / np.sqrt(len(A))
            ax1.plot(it, mu, color=COLOR[m], lw=2.2, label=f"{METHODS[m]} (Adam)")
            ax1.fill_between(it, mu - ci, mu + ci, color=COLOR[m], alpha=0.15, lw=0)
    ax1.axhline(C.ACC_TARGET, color=MUTED, lw=1, ls="--")
    ax1.text(4, C.ACC_TARGET + 0.008, "90% target", ha="left", va="bottom", fontsize=9, color=MUTED)
    ax1.set_title("(a) Adam Optimizer\nDirect overcomes flat landscape via adaptive step", fontsize=11, fontweight="bold")
    ax1.set_xlabel("training update")
    ax1.set_ylabel("test accuracy")
    ax1.set_xlim(0, 320)
    ax1.set_ylim(0.35, 1.0)
    ax1.legend(loc="lower right")

    # Panel 2: Vanilla SGD
    ax2 = axes[1]
    for m in METHODS:
        if (s, m) in runs_sgd:
            A = np.array([r["history"]["test_acc"] for r in runs_sgd[(s, m)]])
            it_sgd = np.array(runs_sgd[(s, m)][0]["history"]["iter"])
            mu = A.mean(0)
            ci = 1.96 * A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros_like(mu)
            ax2.plot(it_sgd, mu, color=COLOR[m], lw=2.2, label=f"{METHODS[m]} (Vanilla SGD)")
            if len(A) > 1 and ci.max() > 0:
                ax2.fill_between(it_sgd, mu - ci, mu + ci, color=COLOR[m], alpha=0.15, lw=0)
    ax2.axhline(C.ACC_TARGET, color=MUTED, lw=1, ls="--")
    ax2.text(4, C.ACC_TARGET + 0.008, "90% target", ha="left", va="bottom", fontsize=9, color=MUTED)
    ax2.set_title("(b) Vanilla Gradient Descent (SGD)\nBarren Plateau halts training: Direct gets trapped flat", fontsize=11, fontweight="bold")
    ax2.set_xlabel("training update")
    ax2.set_ylabel("test accuracy")
    ax2.set_xlim(0, 320)
    ax2.set_ylim(0.35, 1.0)
    ax2.legend(loc="lower right")

    fig.suptitle("Barren Plateau Evidence: Adam vs Vanilla Gradient Descent (Noiseless)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_sgd_figures(log=print):
    C.FIGURES.mkdir(exist_ok=True)
    runs_sgd = load_all_sgd()
    if not runs_sgd:
        log("No SGD runs found in results/runs_sgd, skipping SGD figures.")
        return
    rows_sgd = summarize(runs_sgd)
    write_summary(rows_sgd, C.RESULTS / "summary_sgd.csv")
    log(f"wrote {C.RESULTS / 'summary_sgd.csv'}")
    F = C.FIGURES
    fig_curves_sgd(runs_sgd, F / "fig4_sgd_test_accuracy_curves.png")
    fig_final_accuracy_sgd(rows_sgd, F / "fig5_sgd_final_accuracy.png")
    runs_adam = load_all()
    fig_comparison_adam_vs_sgd(runs_adam, runs_sgd, F / "fig8_adam_vs_sgd_comparison.png")
    log(f"New SGD figures and comparison added to {F} without replacing existing figures.")


def make_all(log=print):
    C.FIGURES.mkdir(exist_ok=True)
    runs = load_all()
    rows = summarize(runs)
    log(f"wrote {write_summary(rows)}")
    F = C.FIGURES
    fig_circuit(F / "fig1_circuit.png")
    fig_training_schemes(F / "fig2_training_schemes.png")
    fig_barren_plateau(F / "fig3_barren_plateau.png")
    fig_curves(runs, F / "fig4_test_accuracy_curves.png")
    fig_final_accuracy(rows, F / "fig5_final_accuracy.png")
    fig_success(rows, F / "fig6_success_rate.png")
    fig_iterations(rows, F / "fig7_iterations_to_converge.png")
    log(f"original figures preserved in {F}")
    make_sgd_figures(log)
    return rows
