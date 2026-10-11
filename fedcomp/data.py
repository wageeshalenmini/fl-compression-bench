"""Datasets and ways of splitting them across devices."""

from __future__ import annotations

import numpy as np


def make_synthetic_classification(m: int, d: int, seed: int = 0):
    """A random linearly-generated binary classification set with labels in {-1,+1}."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, d)) * rng.uniform(0.2, 2.0, size=d)
    x_true = rng.standard_normal(d)
    prob = 1.0 / (1.0 + np.exp(-(A @ x_true)))
    b = np.where(rng.uniform(size=m) < prob, 1.0, -1.0)
    return A, b


def load_libsvm(path: str):
    """Load a LibSVM file (for example a9a) as a dense matrix with labels in {-1,+1}.

    Download the file yourself from the LibSVM datasets page and pass its path.
    """
    from sklearn.datasets import load_svmlight_file

    A, b = load_svmlight_file(path)
    A = np.asarray(A.todense())
    classes = np.unique(b)
    if len(classes) != 2:
        raise ValueError("expected a binary classification dataset")
    return A, np.where(b == classes[1], 1.0, -1.0)


def split_iid(A, b, n: int, seed: int = 0):
    """Shuffle, then cut into n nearly equal parts: every device sees similar data."""
    idx = np.random.default_rng(seed).permutation(len(b))
    parts = np.array_split(idx, n)
    return [A[p] for p in parts], [b[p] for p in parts]


def split_by_label(A, b, n: int):
    """Sort by label, then cut into n parts: devices see very different data."""
    idx = np.argsort(b, kind="stable")
    parts = np.array_split(idx, n)
    return [A[p] for p in parts], [b[p] for p in parts]
