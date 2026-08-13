"""E13: does the hit rate for the canonical optimum survive larger N?

E12 established that the continuous optimum is genuinely unique -- restarts
reaching the best error agree on atom positions to 0.01px, while worse restarts
differ in error as well as position -- and that no mathematical certificate of
having reached it exists (sections 2.2, 10.8, 10.9). What remains is AGREEMENT
BETWEEN INDEPENDENT RESTARTS as an empirical substitute, which worked at N=3 at
a hit rate of about one restart in ten.

That hit rate is the whole procedure's viability. If it decays quickly with N,
the canonical solution is real but unreachable at any budget anyone cares about,
and the practical answer changes completely.

MEASURED WITHOUT KNOWING THE TRUE OPTIMUM. Above N=3 the optimum cannot be
enumerated, but the hit rate does not need it: run R independent restarts,
cluster the results by matched atom distance, and report how many DISTINCT
solutions appear and how large the biggest cluster is. That is the quantity the
procedure actually depends on -- if 60 restarts produce 55 distinct answers,
agreement is not available as a signal regardless of what the true optimum is.

TWO STATISTICS, because one hides the failure mode:
  * the size of the largest cluster, i.e. how often the most-found solution is
    found;
  * whether the LOWEST-ERROR solution lies in that cluster. A procedure that
    reliably converges on something that is not the optimum is worse than one
    that scatters, because it looks canonical and is not.

Matched distance is reported as both median and MAX over the assignment. At
larger N a median hides partial agreement -- seven atoms landing together and
one elsewhere would read as perfect agreement on the median alone.

WHAT THIS CANNOT SHOW. N=8 is still three orders of magnitude below the regime
of interest, and "distinct solutions found in 60 restarts" is a lower bound on
the true number of local optima. A hit rate that survives to 8 does not
establish that it survives to 10^3.
"""

import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from e2_relaxation_gap import _grid, atoms
import e2b_natural as e2b
import e12_canonical_offgrid as e12


def matched(thA, thB, n):
    """Median AND max matched centre distance, in pixels."""
    A, B = thA[:, :2] * n, thB[:, :2] * n
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    r, c = linear_sum_assignment(D)
    d = D[r, c]
    return float(np.median(d)), float(d.max())


def cluster(sols, n, tol=1.0):
    """Greedy clustering by max matched distance. Returns labels."""
    reps, labels = [], []
    for th in sols:
        for k, rth in enumerate(reps):
            if matched(th, rth, n)[1] <= tol:
                labels.append(k)
                break
        else:
            reps.append(th)
            labels.append(len(reps) - 1)
    return np.array(labels), len(reps)


def run(name, n=48, Ns=(3, 4, 6, 8), n_restarts=60, log=print):
    X, Y = _grid(n)
    y = e2b.target(name, n, np.random.default_rng(0))
    half = 0.5 * float(y @ y)
    pc = lambda e: 100.0 * e / half
    log(f"  {name}: {n_restarts} independent continuous restarts per N, "
        f"n={n}")
    log(f"      {'N':>2} {'best':>8} {'distinct':>9} {'largest':>8} "
        f"{'best in':>8} {'hit rate':>9} {'largest':>11} {'2nd cluster':>12}")
    log(f"      {'':>2} {'error':>8} {'solutions':>9} {'cluster':>8} "
        f"{'largest':>8} {'(best)':>9} {'cl. error':>11} {'error':>12}")
    rows = []
    for N in Ns:
        t0 = time.time()
        errs, sols = [], []
        for s in range(n_restarts):
            g = np.random.default_rng(9000 + 137 * N + s)
            th0 = np.column_stack([
                g.uniform(0.05, 0.95, N), g.uniform(0.05, 0.95, N),
                np.log(g.uniform(2.0, 12.0, N)), np.zeros(N),
                np.log(g.uniform(2.0, 12.0, N))])
            A = atoms(th0, X, Y)[0]
            amp0 = np.linalg.lstsq(A.T, y, rcond=None)[0]
            e, th = e12.polish(th0, amp0, y, X, Y, n)
            errs.append(e)
            sols.append(th)
        errs = np.array(errs)
        labels, k = cluster(sols, n)
        ib = int(np.argmin(errs))
        sizes = np.bincount(labels)
        big = int(sizes.argmax())
        # hit rate for the BEST solution specifically
        hit = int((labels == labels[ib]).sum())
        # what the DOMINANT attractor costs -- the number the agreement
        # heuristic would return, against errs.min() which is the right answer
        ebig = float(np.median(errs[labels == big]))
        # and the second-largest, for context on how many such traps there are
        order = np.argsort(-sizes)
        second = order[1] if len(order) > 1 else order[0]
        e2nd = float(np.median(errs[labels == second]))
        log(f"      {N:2d} {pc(errs.min()):8.4f} {k:9d} {int(sizes.max()):8d} "
            f"{'yes' if labels[ib] == big else 'NO':>8} "
            f"{f'{hit}/{n_restarts}':>9} {pc(ebig):10.4f}% {pc(e2nd):11.4f}%"
            f"   [{time.time()-t0:.0f}s]")
        rows.append(dict(N=N, best=float(errs.min()), distinct=k,
                         largest=int(sizes.max()), hit=hit,
                         largest_err=ebig,
                         best_is_largest=bool(labels[ib] == big)))
    return rows


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E13: hit rate for the canonical optimum as N grows.")
    log("# 'distinct solutions' = clusters at 1px max matched distance.")
    log("# 'hit rate (best)' = restarts landing in the same cluster as the")
    log("#   lowest-error solution. This is the number the agreement-based")
    log("#   procedure of section 10.11 depends on.")
    for t in ("cartoon", "ascent"):
        log("")
        run(t, log=log)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
