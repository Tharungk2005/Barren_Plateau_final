"""Real MNIST loading and preprocessing.

Source: OpenML dataset "mnist_784" version 1 (id 554), fetched with
sklearn.datasets.fetch_openml and cached under data/. Rows 0..59999 are the
official MNIST training images and rows 60000..69999 the official test images.
Train/validation images are drawn from the official training part and test
images from the official test part, so the three subsets are disjoint by
construction (and this is additionally asserted on image indices).

No synthetic data is ever generated here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE = DATA_DIR / "mnist_784.npz"
N_OFFICIAL_TRAIN = 60000

# Fixed seed for drawing the data subset. It is the same for every training
# seed and every method, so methods are always compared on identical data.
DATA_SEED = 1234


def load_mnist() -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X uint8 [70000, 784], y int64 [70000], provenance dict)."""
    if not CACHE.exists():
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("mnist_784", version=1, data_home=str(DATA_DIR),
                            as_frame=False, return_X_y=True, parser="liac-arff")
        np.savez_compressed(CACHE, X=X.astype(np.uint8), y=y.astype(np.int64))
    with np.load(CACHE) as f:
        X, y = f["X"], f["y"]
    assert X.shape == (70000, 784) and y.shape == (70000,), "unexpected MNIST shape"
    assert X.max() == 255 and X.min() == 0
    sha = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    prov = {
        "source": "OpenML mnist_784 v1 (id 554) via sklearn.datasets.fetch_openml",
        "cache_file": str(CACHE.relative_to(ROOT)),
        "cache_sha256": sha,
        "shape": list(X.shape),
        "official_train_rows": "0..59999",
        "official_test_rows": "60000..69999",
    }
    return X, y, prov


def downsample(images: np.ndarray, factor: int = 2) -> np.ndarray:
    """Resize 28x28 -> (28/factor)^2 by block averaging; returns floats in [0,1]."""
    n = images.shape[0]
    side = 28 // factor
    im = images.reshape(n, side, factor, side, factor).astype(np.float64) / 255.0
    return im.mean(axis=(2, 4)).reshape(n, side * side)


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    info: dict


def _balanced_indices(y, rows, digits, per_class, rng, exclude=()):
    out = []
    excl = set(int(i) for i in exclude)
    for d in digits:
        cand = np.array([i for i in rows[y[rows] == d] if int(i) not in excl])
        out.append(rng.choice(cand, size=per_class, replace=False))
    return np.concatenate(out)


def make_split(digits=(0, 1), n_features=8, n_train=600, n_val=200, n_test=400,
               angle_range=(0.0, np.pi), data_seed=DATA_SEED) -> Split:
    """Balanced binary MNIST subsets -> downsample -> PCA -> MinMax to angles.

    PCA and the MinMax scaler are fitted on the training subset only.
    Label 1 = digits[1], label 0 = digits[0].
    """
    X, y, prov = load_mnist()
    rng = np.random.default_rng(data_seed)
    tr_rows = np.arange(N_OFFICIAL_TRAIN)
    te_rows = np.arange(N_OFFICIAL_TRAIN, len(y))
    idx_tr = _balanced_indices(y, tr_rows, digits, n_train // 2, rng)
    idx_va = _balanced_indices(y, tr_rows, digits, n_val // 2, rng, exclude=idx_tr)
    idx_te = _balanced_indices(y, te_rows, digits, n_test // 2, rng)
    # disjointness checks (indices and official partition)
    assert len(set(idx_tr) & set(idx_va)) == 0
    assert len(set(idx_tr) & set(idx_te)) == 0 and len(set(idx_va) & set(idx_te)) == 0
    assert idx_te.min() >= N_OFFICIAL_TRAIN and max(idx_tr.max(), idx_va.max()) < N_OFFICIAL_TRAIN
    # shuffle order within each subset (deterministic)
    idx_tr, idx_va, idx_te = (rng.permutation(a) for a in (idx_tr, idx_va, idx_te))

    small = {k: downsample(X[i]) for k, i in (("tr", idx_tr), ("va", idx_va), ("te", idx_te))}
    pca = PCA(n_components=n_features, random_state=0).fit(small["tr"])
    feats = {k: pca.transform(v) for k, v in small.items()}
    scaler = MinMaxScaler(feature_range=angle_range).fit(feats["tr"])
    ang = {k: np.clip(scaler.transform(v), *angle_range) for k, v in feats.items()}
    lab = {k: (y[i] == digits[1]).astype(np.float64)
           for k, i in (("tr", idx_tr), ("va", idx_va), ("te", idx_te))}

    info = {
        "provenance": prov,
        "digits": list(digits),
        "label_map": {str(digits[0]): 0, str(digits[1]): 1},
        "n_features": n_features,
        "preprocessing": "28x28 -> 14x14 block-average, /255, PCA(fit train) -> "
                         f"{n_features} comps, MinMax(fit train) to [{angle_range[0]:.4f}, "
                         f"{angle_range[1]:.4f}] rad, clipped",
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "sizes": {"train": len(idx_tr), "val": len(idx_va), "test": len(idx_te)},
        "class_counts": {k: {str(d): int((y[i] == d).sum()) for d in digits}
                         for k, i in (("train", idx_tr), ("val", idx_va), ("test", idx_te))},
        "data_seed": data_seed,
        "index_sha256": hashlib.sha256(
            json.dumps([idx_tr.tolist(), idx_va.tolist(), idx_te.tolist()]).encode()).hexdigest(),
    }
    return Split(ang["tr"], lab["tr"], ang["va"], lab["va"], ang["te"], lab["te"], info)


def raw_images(digits, n_train=600, n_val=200, n_test=400, data_seed=DATA_SEED):
    """The original 28x28 images of the same subsets (for display only)."""
    X, y, _ = load_mnist()
    rng = np.random.default_rng(data_seed)
    tr_rows = np.arange(N_OFFICIAL_TRAIN)
    idx_tr = _balanced_indices(y, tr_rows, digits, n_train // 2, rng)
    _balanced_indices(y, tr_rows, digits, n_val // 2, rng, exclude=idx_tr)
    _balanced_indices(y, np.arange(N_OFFICIAL_TRAIN, len(y)), digits, n_test // 2, rng)
    idx_tr = rng.permutation(idx_tr)
    return X[idx_tr].reshape(-1, 28, 28), y[idx_tr]
