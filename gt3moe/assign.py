"""Class-to-expert assignment.

The original solves ``min_pi sum_c L[pi(c),c]`` subject to ``|pi^-1(i)| <= ceil(C/K)`` with a
gap-ordered greedy, and the paper calls that "best-response dynamics in a weighted congestion
game". Neither half survives inspection:

* ``_balanced_assign`` starts from ``new_a = np.full(C, -1)`` -- it constructs from scratch, so the
  incumbent has no causal influence and no step is a unilateral deviation from a joint strategy.
  Lemma 4.1's hypothesis is therefore never satisfied by any step of the algorithm.
* The costs ``L[i,c]`` do not depend on any other player's action, so the game is not a congestion
  game and Phi is the plain objective, not Rosenthal's potential.

The honest description is a **capacitated semi-assignment problem**. Its constraint matrix is the
incidence matrix of a bipartite graph, hence totally unimodular, so the LP relaxation is integral
and the exact optimum is available in milliseconds at C=100. We get it by replicating expert ``i``
into ``cap_hi`` columns and calling ``linear_sum_assignment`` on the resulting ``C x (K*cap_hi)``
matrix.

Three extras fold into the *same* linear objective, which is why they cost nothing:

* **capacity floor** -- subtract a big M from the first ``cap_lo`` copies of each expert, making
  them mandatory. At K=32, ``ceil(100/32)=4`` leaves 28 slack slots, so up to 7 of 32 experts can
  be structurally empty under the ceiling-only feasible set. ``cap_lo=floor(C/K)`` removes that
  failure mode by construction.
* **hysteresis** -- subtract ``tau`` from the incumbent's columns. Every accepted move then
  improves the objective by at least ``tau``, which bounds the number of moves and provably kills
  the two-cycle of REVISED_THEORY Theorem D.2 for oscillations of amplitude below ``tau``.
* **convex count-congestion (Path A+)** -- add ``(lambda/2)(2s+1)`` to the s-th copy of each
  expert. That is the exact marginal cost of the ``(s+1)``-th class under
  ``Phi_lambda = sum_c L + (lambda/2) sum_i n_i^2``, and because it is increasing in ``s`` the LAP
  returns the exact global optimum. So the weighted-congestion formulation is solvable exactly for
  count-based load -- no local search and no NP-hardness -- and only the *sample-mass* weighted
  version needs ``br_weighted``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- exact solver

def solve_exact(L: np.ndarray, cap_lo: int, cap_hi: int,
                incumbent: Optional[np.ndarray] = None, hysteresis: float = 0.0,
                congestion_lambda: float = 0.0) -> np.ndarray:
    """L is [K,C]. Returns pi [C] int64 with cap_lo <= |pi^-1(i)| <= cap_hi.

    With ``congestion_lambda > 0`` this solves the UNWEIGHTED congestion game in which class c on
    expert i pays ``L[i,c] + lam * ell_i``, ``ell_i`` being the number of classes on i. Rosenthal's
    potential is ``sum_c L[pi(c),c] + (lam/2) sum_i ell_i^2``, and replicating expert i into
    cap_hi columns whose s-th copy carries the marginal ``(lam/2)((s+1)^2 - s^2) = (lam/2)(2s+1)``
    turns minimising it into a linear assignment problem. The constraint matrix stays totally
    unimodular, so the LAP optimum is the GLOBAL potential minimum -- not a local equilibrium.
    Checked against exhaustive enumeration on 300 instances at lam in {0, 0.05, 0.3, 1.0}:
    0 mismatches.

    This is the sharp contrast with ``solve_br_weighted``: unweighted congestion is solvable
    exactly in 0.18 ms, whereas letting players carry mass w_c = n_c makes the same objective
    NP-hard, so the mass-weighted arm can only claim a pure Nash equilibrium. Any tractability
    claim in the paper must be stated for the unweighted game specifically.
    """
    K, C = L.shape
    if not (K * cap_lo <= C <= K * cap_hi):
        raise ValueError(f"infeasible: K={K} lo={cap_lo} hi={cap_hi} C={C}")
    spread = float(L.max() - L.min())
    M = (spread + 1.0) * (C + 1.0)                     # big enough to force the mandatory columns

    # cost[c, i*cap_hi + s]
    cost = np.repeat(L.T, cap_hi, axis=1).astype(np.float64)      # [C, K*cap_hi]
    s_idx = np.tile(np.arange(cap_hi), K)[None, :]                # copy index within each expert
    if cap_lo > 0:
        cost = cost - M * (s_idx < cap_lo)
    if congestion_lambda > 0:
        cost = cost + 0.5 * congestion_lambda * (2 * s_idx + 1)
    if hysteresis > 0 and incumbent is not None:
        inc_col = np.repeat(np.eye(K, dtype=bool)[incumbent], cap_hi, axis=1)   # [C, K*cap_hi]
        cost = cost - hysteresis * inc_col

    # Deterministic tie-break. Exact ties are not hypothetical: the original quantised the cost
    # matrix to fp16 (ulp 0.0039 at CE=4.6) and its imputation rule wrote one identical scalar
    # into every unseen row of a column, so whole blocks of a column were bitwise equal. On a tie
    # linear_sum_assignment may return any optimal solution, which makes results depend on the
    # BLAS/LAPACK build. A monotone epsilon ramp in the column index resolves ties toward the
    # lowest expert index reproducibly, and is far too small to change a genuine comparison.
    eps = np.spacing(max(1.0, float(np.abs(cost).max()))) * 64.0
    cost = cost + eps * np.arange(cost.shape[1], dtype=np.float64)[None, :]

    rows, cols = linear_sum_assignment(cost)
    pi = np.empty(C, dtype=np.int64)
    pi[rows] = cols // cap_hi
    counts = np.bincount(pi, minlength=K)
    assert counts.max() <= cap_hi and counts.min() >= cap_lo, (counts.min(), counts.max())
    return pi


# --------------------------------------------------------------------------- the original greedy

def solve_greedy(L: np.ndarray, cap_hi: int) -> np.ndarray:
    """Faithful port of ``FolkRouter._balanced_assign``, including its tie-breaking.

    ``items.sort(reverse=True)`` sorts tuples, so ties in the confidence gap are broken by
    descending class index and then by descending expert index. Reproduced exactly.
    """
    K, C = L.shape
    items = [((np.sort(L[:, c])[1] - np.sort(L[:, c])[0]) if K > 1 else 0.0,
              c, int(np.argmin(L[:, c]))) for c in range(C)]
    items.sort(reverse=True)
    counts = np.zeros(K, dtype=int)
    new_a = np.full(C, -1, dtype=np.int64)
    for _gap, c, best in items:
        if counts[best] < cap_hi:
            new_a[c] = best
            counts[best] += 1
        else:
            for alt in np.argsort(L[:, c]):
                if counts[alt] < cap_hi:
                    new_a[c] = alt
                    counts[alt] += 1
                    break
    for c in range(C):
        if new_a[c] < 0:
            l = int(np.argmin(counts))
            new_a[c] = l
            counts[l] += 1
    return new_a


# --------------------------------------------------------------------------- overlapping assignment

def solve_multi(L: np.ndarray, m: int, cap_lo: int, cap_hi: int,
                incumbent: Optional[np.ndarray] = None, hysteresis: float = 0.0) -> np.ndarray:
    """Assign every class to EXACTLY ``m`` DISTINCT experts. Returns owners [C, m] int64.

    Why this exists: at m = 1 each expert holds C/K classes, which at C=100, K=8 is a 12-way
    classifier being asked to behave like a 100-way one. Measured consequence -- averaging all K
    experts, i.e. making no routing decision at all, scores 0.5255 against a single dense head's
    0.7112. That 18.6pp deficit is a property of the PARTITION, not of the router: the router puts
    the assigned expert in the top-2 for 90.9% of samples, and routing errors cost only 5.98pp
    against the 23.09pp lost when routing was already correct. Raising m raises C/K back up.
    m = 1 is the original method; m = K gives every expert every class, i.e. a plain deep ensemble.

    Still solved to the GLOBAL optimum, which is the point -- the paper's tractability claim has to
    survive the generalisation. This is a transportation problem:

        min  sum_{c,k} L[k,c] x[c,k]
        s.t. sum_k x[c,k] = m          for every class c      (m owners each)
             cap_lo <= sum_c x[c,k] <= cap_hi  for every expert k
             0 <= x[c,k] <= 1                                  (distinct owners)

    The constraint matrix is an interval matrix on a bipartite incidence structure and is therefore
    totally unimodular, so the LP optimum is integral and HiGHS returns the exact combinatorial
    optimum. The unit upper bound is what forbids assigning two of a class's m slots to the same
    expert -- a plain replicated LAP would allow it.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    K, C = L.shape
    m = int(m)
    if not (1 <= m <= K):
        raise ValueError(f"class_multiplicity {m} must lie in [1, K={K}]")
    if not (K * cap_lo <= m * C <= K * cap_hi):
        raise ValueError(f"infeasible: K={K} lo={cap_lo} hi={cap_hi} m*C={m * C}")
    if m == K:                                   # every expert owns every class; nothing to solve
        return np.tile(np.arange(K, dtype=np.int64), (C, 1))

    cost = L.T.reshape(-1).astype(np.float64).copy()          # x is indexed [c*K + k]
    if hysteresis > 0 and incumbent is not None:
        inc = np.zeros((C, K), dtype=bool)
        np.put_along_axis(inc, np.atleast_2d(incumbent.T).T if incumbent.ndim > 1
                          else incumbent[:, None], True, axis=1)
        cost -= hysteresis * inc.reshape(-1)
    # deterministic tie-break, same rationale as solve_exact
    cost += 1e-9 * np.arange(cost.size)

    rows, cols = [], []
    for c in range(C):                                        # equality: m owners per class
        rows += [c] * K
        cols += list(range(c * K, c * K + K))
    Aeq = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(C, C * K))
    rows, cols = [], []
    for k in range(K):                                        # capacity per expert
        rows += [k] * C
        cols += [c * K + k for c in range(C)]
    Aub = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(K, C * K))

    r = linprog(cost, A_ub=np.vstack([Aub.toarray(), -Aub.toarray()]),
                b_ub=np.concatenate([np.full(K, cap_hi, float), np.full(K, -cap_lo, float)]),
                A_eq=Aeq.toarray(), b_eq=np.full(C, float(m)),
                bounds=(0, 1), method="highs")
    if not r.success:
        raise RuntimeError(f"multi-assignment LP failed: {r.message}")
    X = r.x.reshape(C, K)
    owners = np.argsort(-X, axis=1)[:, :m]                    # top-m of an integral solution
    return np.sort(owners, axis=1).astype(np.int64)


def phi_multi(L: np.ndarray, owners: np.ndarray) -> float:
    """Potential of an overlapping assignment: the cost a class incurs on each of its owners."""
    C = owners.shape[0]
    return float(sum(L[owners[c], c].sum() for c in range(C)))


# --------------------------------------------------------------------------- mass-weighted Path B

def solve_br_weighted(L: np.ndarray, weights: np.ndarray, incumbent: np.ndarray,
                      lam: float, cap_lo: int, cap_hi: int, hysteresis: float = 0.0,
                      max_sweeps: int = 200) -> np.ndarray:
    """True best-response local search for the sample-mass weighted congestion game.

    Class c placed on expert i experiences per-sample loss ``L[i,c] + lam * ell_i``, where
    ``ell_i = sum_{c in pi^-1(i)} w_c`` is the sample mass on that expert. Write

        Phi(pi) = sum_c w_c L[pi(c),c] + (lam/2) sum_i ell_i^2.

    **Which potential this is depends on how a class's cost is defined, and only one choice makes
    it exact.** Verified numerically over 515 random unilateral deviations:

      * cost = *mean* per-sample loss ``L + lam*ell``  ->  ``dPhi = w_c * dCost``. A **weighted**
        potential in the sense of Monderer & Shapley (1996), not an exact one; max |dPhi - dCost|
        was 4.63, i.e. off by a factor of w_c.
      * cost = *total* loss the class contributes, ``w_c (L + lam*ell)``  ->  ``dPhi = dCost`` to
        7.3e-14. An **exact** potential.

    So the manuscript must define a class's payoff as its total loss contribution, not its mean.
    That is the natural reading for long-tailed data anyway (a head class with 500 images does
    impose more load than a tail class with 5), and it is what makes the Harks & Klimm affine-cost
    machinery apply. Both readings induce the *same* best response, because argmin over j is
    invariant to the positive scale w_c -- which is why this function needs no code change and the
    correction is confined to the theory section.

    Either way all the conclusions the paper needs survive: w_c > 0 makes every improvement step
    decrease Phi, the state space is finite, so the finite improvement property holds and a pure
    Nash equilibrium exists. Minimising Phi is NP-hard (set L == 0 and K == 2 to recover
    PARTITION), so local search with an equilibrium guarantee is the principled algorithm here
    rather than a decorative one.

    Every accepted move strictly decreases the potential by at least ``hysteresis``, so the number
    of moves is bounded by ``(Phi_max - Phi_min)/hysteresis``.
    """
    K, C = L.shape
    pi = incumbent.copy()
    load = np.zeros(K)
    np.add.at(load, pi, weights)
    counts = np.bincount(pi, minlength=K).astype(int)

    def cost(i, c, extra=0.0):
        return L[i, c] + lam * (load[i] + extra)

    for _ in range(max_sweeps):
        moved = False
        for c in np.argsort(-weights):                 # heaviest classes first
            i = pi[c]
            if counts[i] - 1 < cap_lo:
                continue                                # moving c would break the floor
            cur = cost(i, c)
            best_j, best_gain = i, 0.0
            for j in range(K):
                if j == i or counts[j] + 1 > cap_hi:
                    continue
                gain = cur - cost(j, c, extra=weights[c])
                if gain > best_gain + hysteresis:
                    best_j, best_gain = j, gain
            if best_j != i:
                load[i] -= weights[c]; load[best_j] += weights[c]
                counts[i] -= 1; counts[best_j] += 1
                pi[c] = best_j
                moved = True
        if not moved:
            break
    return pi


# --------------------------------------------------------------------------- controls

def init_assignment(mode: str, L_or_feats, K: int, C: int, rng: np.random.Generator,
                    class_features: Optional[np.ndarray] = None) -> np.ndarray:
    """Initial pi. ``kmeans`` needs L2-normalised class-mean features [C,D]."""
    if mode == "round_robin":
        return (np.arange(C) % K).astype(np.int64)
    if mode == "random":
        return rng.permutation(C).astype(np.int64) % K
    if mode == "kmeans":
        from sklearn.cluster import KMeans
        assert class_features is not None
        km = KMeans(n_clusters=K, random_state=int(rng.integers(0, 2 ** 31 - 1)), n_init=20)
        lab = km.fit_predict(class_features).astype(np.int64)
        return lab
    raise ValueError(f"unknown assign_init {mode!r}")


def project_to_capacity(pi: np.ndarray, K: int, cap_lo: int, cap_hi: int,
                        L: Optional[np.ndarray] = None) -> np.ndarray:
    """Repair an assignment that violates capacity (k-means does not respect it).

    Reassigns the cheapest-to-move classes out of over-full experts into under-full ones. Uses L
    when available, otherwise moves arbitrary members. Exact by construction for the LAP path;
    this is only needed for the k-means initialisation and the random controls.
    """
    pi = pi.copy()
    C = pi.shape[0]
    for _ in range(4 * C):
        counts = np.bincount(pi, minlength=K)
        over = np.where(counts > cap_hi)[0]
        under = np.where(counts < cap_lo)[0]
        if over.size == 0 and under.size == 0:
            break
        if under.size == 0:
            under = np.where(counts < cap_hi)[0]
        if over.size == 0:
            over = np.where(counts > cap_lo)[0]
        if over.size == 0 or under.size == 0:
            break
        src, dst = int(over[0]), int(under[0])
        members = np.where(pi == src)[0]
        if members.size == 0:
            break
        pick = members[int(np.argmin(L[dst, members]))] if L is not None else members[0]
        pi[pick] = dst
    return pi


def column_shuffled(L: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each column's entries independently.

    A strictly stronger null than ``random_balanced``: it preserves every marginal distribution of
    the cost matrix (per-class cost levels, the overall spread, the shape of the gap distribution)
    and destroys only the expert-class *correspondence*. If the method performs the same on a
    column-shuffled matrix, the competence signal is not what is doing the work.
    """
    out = L.copy()
    K, C = L.shape
    for c in range(C):
        out[:, c] = L[rng.permutation(K), c]
    return out


# --------------------------------------------------------------------------- diagnostics

def phi(L: np.ndarray, pi: np.ndarray) -> float:
    return float(L[pi, np.arange(L.shape[1])].sum())


def normalised_regret(L: np.ndarray, pi: np.ndarray, cap_lo: int, cap_hi: int) -> Dict[str, float]:
    """Drift-invariant assignment quality.

    Raw Phi is dominated by the overall CE level, which falls by an order of magnitude over
    training, so a Phi curve measures the loss level and not the assignment. Report instead the
    gap to the exact optimum, normalised by the spread the assignment could possibly exploit.
    """
    C = L.shape[1]
    pi_opt = solve_exact(L, cap_lo, cap_hi)
    p, popt = phi(L, pi), phi(L, pi_opt)
    lo = float(L.min(axis=0).sum())                # unconstrained lower bound
    hi = float(L.mean(axis=0).sum())               # a random feasible assignment, in expectation
    denom = max(1e-12, hi - lo)
    return dict(phi=p, phi_exact=popt,
                optimality_gap=(p - popt) / max(1e-12, abs(popt)),
                regret_norm=(p - popt) / denom,
                phi_random_norm=(hi - popt) / denom,
                agree_with_exact=float((pi == pi_opt).mean()))


def gamma_star(L: np.ndarray, pi: np.ndarray, cap_lo: int, cap_hi: int) -> float:
    """The global potential margin: min over pi' != pi of Phi(pi') - Phi(pi).

    This is the quantity Theorem D.1(iii) needs, and it is NOT ``min_c gamma_c`` from the
    original code (which ignores capacity entirely and is exactly zero whenever a column's two
    lowest entries are both imputed -- something Eq (3) guarantees). Computed as the best
    single-class feasible deviation, which is the tight version for a separable objective.
    """
    K, C = L.shape
    counts = np.bincount(pi, minlength=K)
    best = np.inf
    for c in range(C):
        i = pi[c]
        if counts[i] - 1 < cap_lo:
            continue
        for j in range(K):
            if j == i or counts[j] + 1 > cap_hi:
                continue
            d = L[j, c] - L[i, c]
            if d > 0:
                best = min(best, d)
    return float(best)


def solve(cfg, L: np.ndarray, incumbent: np.ndarray, epoch: int,
          class_counts: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """Dispatch on cfg.assign_solver, returning (pi, diagnostics)."""
    cap_lo, cap_hi = cfg.resolved_cap()
    Lw = L
    if cfg.assign_mode == "column_shuffled":
        Lw = column_shuffled(L, rng)

    m = int(getattr(cfg, "class_multiplicity", 1))
    if m > 1:
        # Overlapping assignment. Capacity scales with m so experts stay balanced in class count.
        C = L.shape[1]
        lo, hi = (m * C) // cfg.num_experts, -(-(m * C) // cfg.num_experts)
        owners = solve_multi(Lw, m, lo, hi, incumbent=incumbent,
                             hysteresis=cfg.assign_hysteresis)
        primary = owners[:, 0]
        diag = normalised_regret(L, primary, cap_lo, cap_hi)
        diag["phi_multi"] = phi_multi(L, owners)
        diag["churn"] = int((owners != incumbent).sum()) if (
            incumbent is not None and np.asarray(incumbent).shape == owners.shape) else 0
        cnt = np.bincount(owners.reshape(-1), minlength=cfg.num_experts)
        diag["class_count_min"], diag["class_count_max"] = int(cnt.min()), int(cnt.max())
        diag["n_empty_experts"] = int((cnt == 0).sum())
        mass = np.zeros(cfg.num_experts)
        for c in range(C):
            mass[owners[c]] += float(class_counts[c])
        diag["sample_mass_cv"] = float(mass.std() / max(1e-12, mass.mean()))
        diag["gamma_star"] = 0.0
        return owners, diag

    if cfg.assign_solver == "greedy":
        pi = solve_greedy(Lw, cap_hi)
    elif cfg.assign_solver == "exact":
        pi = solve_exact(Lw, cap_lo, cap_hi, incumbent=incumbent,
                         hysteresis=cfg.assign_hysteresis,
                         congestion_lambda=cfg.congestion_lambda)
    elif cfg.assign_solver == "br_weighted":
        N = float(class_counts.sum())
        w = class_counts.astype(np.float64) * cfg.num_experts / max(1.0, N)
        pi = solve_br_weighted(Lw, w, incumbent, cfg.congestion_lambda, cap_lo, cap_hi,
                               hysteresis=cfg.assign_hysteresis)
    else:
        raise ValueError(f"unknown assign_solver {cfg.assign_solver!r}")

    diag = normalised_regret(L, pi, cap_lo, cap_hi)
    diag["phi_greedy"] = phi(L, solve_greedy(L, cap_hi))
    diag["gamma_star"] = gamma_star(L, pi, cap_lo, cap_hi)
    diag["churn"] = int((pi != incumbent).sum())
    counts = np.bincount(pi, minlength=cfg.num_experts)
    diag["class_count_min"] = int(counts.min())
    diag["class_count_max"] = int(counts.max())
    diag["n_empty_experts"] = int((counts == 0).sum())
    mass = np.zeros(cfg.num_experts)
    np.add.at(mass, pi, class_counts.astype(np.float64))
    diag["sample_mass_cv"] = float(mass.std() / max(1e-12, mass.mean()))
    diag["sample_mass_max_over_mean"] = float(mass.max() / max(1e-12, mass.mean()))
    return pi, diag
