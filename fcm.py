"""
Fuzzy C-Means (Bezdek, 1981), written from scratch with NumPy.

Minimises  J_m = sum_i sum_k  u_ik^m * ||x_k - v_i||^2
subject to sum_i u_ik = 1  (every pixel's memberships across clusters add up to 1)

Alternating updates:
    u_ik = 1 / sum_j ( d_ik / d_jk )^(2/(m-1))         membership of pixel k in cluster i
    v_i  = sum_k u_ik^m x_k / sum_k u_ik^m               cluster centre

m > 1 is the fuzzifier: m -> 1 gives crisp k-means, larger m gives softer boundaries.
"""

import numpy as np

EPS = 1e-10


def distances(X, V):
    """Squared Euclidean distance of every point to every centre: (c, N)."""
    return ((X[None, :, :] - V[:, None, :]) ** 2).sum(-1) + EPS


def memberships(X, V, m):
    d2 = distances(X, V)
    p = 1.0 / (m - 1.0)
    inv = d2 ** -p                       # (d_ik^2)^(-1/(m-1))
    return inv / inv.sum(0, keepdims=True)


def objective(X, V, U, m):
    return float(((U ** m) * distances(X, V)).sum())


def reformulated_objective(X, V, m):
    """Hathaway & Bezdek (1995): FCM objective with the optimal memberships for
    centres V substituted in. Depends on V only, so a GA can use it as fitness:
        R_m(V) = sum_k ( sum_i d_ik^(2/(1-m)) )^(1-m)"""
    d2 = distances(X, V)
    return float(((d2 ** (1.0 / (1.0 - m))).sum(0) ** (1.0 - m)).sum())


def fcm(X, c, m=2.0, init_centers=None, max_iter=300, tol=1e-5, seed=None):
    """Returns centres V (c, f), memberships U (c, N), objective history, iterations."""
    rng = np.random.default_rng(seed)
    if init_centers is None:
        V = X[rng.choice(len(X), c, replace=False)].copy()      # random data points (Forgy)
    else:
        V = np.array(init_centers, dtype=float)
    history = []
    for it in range(1, max_iter + 1):
        U = memberships(X, V, m)
        Um = U ** m
        V_new = (Um @ X) / Um.sum(1, keepdims=True)
        history.append(objective(X, V_new, U, m))
        shift = np.abs(V_new - V).max()
        V = V_new
        if shift < tol:
            break
    U = memberships(X, V, m)
    return V, U, history, it


def partition_entropy(U):
    """Per-point normalised entropy of the membership vector, 0 (certain) .. 1 (maximally fuzzy)."""
    c = U.shape[0]
    return -(U * np.log(U + EPS)).sum(0) / np.log(c)


def partition_coefficient(U):
    """Bezdek's validity index: 1/N sum u^2. 1 = crisp, 1/c = completely fuzzy."""
    return float((U ** 2).sum(0).mean())
