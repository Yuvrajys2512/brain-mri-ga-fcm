"""Plain (crisp) k-means with random initialisation: the baseline. Every pixel belongs to exactly one cluster."""

import numpy as np

from fcm import distances


def kmeans(X, k, max_iter=300, tol=1e-6, seed=None):
    rng = np.random.default_rng(seed)
    V = X[rng.choice(len(X), k, replace=False)].copy()
    for it in range(1, max_iter + 1):
        labels = distances(X, V).argmin(0)
        V_new = np.array([X[labels == j].mean(0) if np.any(labels == j) else V[j] for j in range(k)])
        shift = np.abs(V_new - V).max()
        V = V_new
        if shift < tol:
            break
    labels = distances(X, V).argmin(0)
    sse = float(((X - V[labels]) ** 2).sum())
    return V, labels, sse, it
