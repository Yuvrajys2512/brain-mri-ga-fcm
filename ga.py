"""
Real-coded Genetic Algorithm that searches for good initial FCM cluster centres.

Chromosome   c centres x f features, flattened (real values in [0, 1])
Fitness      Hathaway-Bezdek reformulated FCM objective R_m(V) (lower = better):
             the FCM objective you would get with these centres and optimal memberships
Selection    tournament (size 3)
Crossover    BLX-alpha blend crossover
Mutation     Gaussian, per gene, clipped to [0, 1]
Elitism      the best individuals survive unchanged

Centres are sorted by their first feature (intensity) before evaluation, so the
same partition always has the same encoding (removes label-permutation symmetry).
"""

from dataclasses import dataclass, field

import numpy as np

from fcm import reformulated_objective


@dataclass
class GAConfig:
    pop_size: int = 30
    generations: int = 40
    tournament: int = 3
    crossover_rate: float = 0.9
    blx_alpha: float = 0.5
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.08
    elite: int = 2
    sample: int = 3000          # pixels used to evaluate fitness (speed)


@dataclass
class GAResult:
    centers: np.ndarray
    fitness: float
    best_history: list = field(default_factory=list)
    mean_history: list = field(default_factory=list)
    evaluations: int = 0


def _canonical(chrom, c, f):
    V = chrom.reshape(c, f)
    return V[np.argsort(V[:, 0])]


def ga_init_centers(X, c, m=2.0, cfg=GAConfig(), seed=None):
    rng = np.random.default_rng(seed)
    Xs = X if len(X) <= cfg.sample else X[rng.choice(len(X), cfg.sample, replace=False)]
    f = X.shape[1]
    lo, hi = X.min(0), X.max(0)
    n_genes = c * f

    def fitness(chrom):
        return reformulated_objective(Xs, _canonical(chrom, c, f), m)

    # initial population: half uniform random in the data range, half random data points
    pop = rng.uniform(np.tile(lo, c), np.tile(hi, c), size=(cfg.pop_size, n_genes))
    half = cfg.pop_size // 2
    pop[:half] = Xs[rng.integers(0, len(Xs), size=(half, c))].reshape(half, n_genes)
    fit = np.array([fitness(p) for p in pop])
    evals = len(pop)
    res = GAResult(None, None)

    for _ in range(cfg.generations):
        order = np.argsort(fit)
        res.best_history.append(float(fit[order[0]]))
        res.mean_history.append(float(fit.mean()))
        new = [pop[i].copy() for i in order[:cfg.elite]]                     # elitism

        def select():
            idx = rng.integers(0, cfg.pop_size, cfg.tournament)
            return pop[idx[np.argmin(fit[idx])]]

        while len(new) < cfg.pop_size:
            a, b = select(), select()
            if rng.random() < cfg.crossover_rate:                             # BLX-alpha
                lo_g, hi_g = np.minimum(a, b), np.maximum(a, b)
                span = hi_g - lo_g
                child = rng.uniform(lo_g - cfg.blx_alpha * span, hi_g + cfg.blx_alpha * span)
            else:
                child = a.copy()
            mask = rng.random(n_genes) < cfg.mutation_rate                    # Gaussian mutation
            child[mask] += rng.normal(0, cfg.mutation_sigma, mask.sum())
            new.append(np.clip(child, np.tile(lo, c), np.tile(hi, c)))
        pop = np.array(new)
        fit = np.array([fitness(p) for p in pop])
        evals += len(pop)

    best = int(np.argmin(fit))
    res.best_history.append(float(fit[best]))
    res.mean_history.append(float(fit.mean()))
    res.centers = _canonical(pop[best], c, f)
    res.fitness = float(fit[best])
    res.evaluations = evals
    return res
