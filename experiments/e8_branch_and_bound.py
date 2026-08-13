"""E8: exact l0 branch-and-bound -- a certified global optimum at section 10's budgets.

Section 2.2 rules out measure-space convexification as a route to (P0): the
closed convex hull of the N-sparse measures of bounded mass is the whole TV
ball for every N, so no convex penalty on measures can see the atom count. Its
positive half is that on a FINITE dictionary with bounded amplitudes the hull
does depend on N, and that is what exact l0 branch-and-bound exploits. E4
confirmed the cost of ignoring this -- exact l1 loses to exact l0 by 6-409% --
but could only reach N<=4, because exhaustive enumeration is C(D,N).

This closes that gap. It is the only route in this repository to a CERTIFIED
GLOBAL optimum rather than a bound, at the budgets section 10 actually reports.

FORMULATION (big-M, the standard one). Minimise 0.5||y - Gc||^2 subject to
||c||_0 <= N and ||c||_inf <= M. The box is not cosmetic: with a dictionary of
coherence 0.985 an unconstrained support can be a near-duplicate pair with
enormous opposing amplitudes -- correct arithmetic, useless as an encoding, and
the thing that makes every relaxation vacuous. Every result below is therefore
a global optimum OVER THE BOX, with M reported and swept.

THE NODE BOUND. At a node with atoms I forced in, let r be the least-squares
residual on I and R = ||r||^2. Since r is orthogonal to span(G_I), for any t the
vector p = t*r is admissible in the Fenchel/Hoelder bound of section 10.2, and
the forced-in atoms contribute nothing to <p, Gc>. With k = N - |I| atoms still
to place, each of amplitude at most M,

    0.5||y - Gc||^2  >=  t*R - 0.5*t^2*R - t*M*sigma,    sigma = sum of the k
                                                          largest |<r,g_j>|
                                                          over the free atoms

which maximises at t = 1 - M*sigma/R, giving

    LB  =  (R - M*sigma)_+^2 / (2R).                                      (*)

Two remarks. The weaker bound 0.5*R - M*sigma, obtained by discarding the
quadratic term, is dominated by (*) for every input -- their difference is
R*x^2/2 with x = M*sigma/R -- so only (*) is computed. And (*) is exactly
section 10.2's instrument specialised to a node, which is why the box matters
so much here: sigma is multiplied by M, so a loose box makes the bound vacuous
and the tree explodes.

COST. Everything after setup runs on the Gram matrix and b = G y, never on
pixels: <r,g_j> = b_j - sum_i c_i Gram[j,i], and ||r||^2 = ||y||^2 - c.b_I at a
least-squares point. So a node costs O(D|I|) rather than O(DP), which is what
makes 1e5-1e6 nodes affordable.

ANYTIME. Best-first search on the bound means the front of the queue IS the
global lower bound at every instant. If the queue empties the incumbent is
proven optimal; if the node cap is hit, the gap (incumbent - queue front) is a
certified bound on how far the incumbent can be from the global optimum. Either
way a number comes out, which is the property section 10.2's bound lacked.

WHAT THIS CAN SHOW. That greedy is or is not optimal on a given dictionary at a
given N, and if not, by exactly how much.

WHAT THIS CANNOT SHOW. Anything off-grid: this is the global optimum over a
finite dictionary, and E5 showed off-grid refinement moves errors far more than
dictionary choice does. Nor anything about N ~ 1e3. A certified gap of 0% means
optimal ON THIS DICTIONARY UNDER THIS BOX, nothing wider.
"""

import heapq
import sys
import time

import numpy as np

from e2_relaxation_gap import _grid
import e2b_natural as e2b
import e4_exact_l0 as e4


# ------------------------------------------------------------------ incumbent
def solve_support(Gram, b, S, M):
    """Least squares on support S, projected to the box if it escapes.

    The unconstrained solution is optimal for the box problem whenever it
    already satisfies the box, which is the common case for a generous M; only
    the rest pay for a small bounded QP."""
    S = list(S)
    A, bb = Gram[np.ix_(S, S)], b[S]
    try:
        c = np.linalg.solve(A + 1e-12 * np.eye(len(S)), bb)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, bb, rcond=None)[0]
    if np.abs(c).max() <= M:
        return c, float(c @ bb)                       # explained energy
    from scipy.optimize import minimize
    f = lambda z: (0.5 * z @ A @ z - z @ bb, A @ z - bb)
    r = minimize(f, np.clip(c, -M, M), jac=True, method="L-BFGS-B",
                 bounds=[(-M, M)] * len(S))
    c = r.x
    return c, float(2 * (c @ bb) - c @ A @ c)


def greedy_then_swap(Gram, b, N, M, max_rounds=40):
    """OMP, then exhaustive single-atom swaps until no swap improves.

    A strong incumbent is what makes the tree small, so this is worth the cost:
    every improvement here prunes exponentially many nodes later."""
    D = len(b)
    S = []
    for _ in range(N):
        if S:
            c, _ = solve_support(Gram, b, S, M)
            resid_corr = b - Gram[:, S] @ c
        else:
            resid_corr = b.copy()
        # mask AFTER taking magnitudes: masking with -inf first and then
        # applying abs turns it into +inf, so the argmax re-selects an atom
        # already in S and the support fills with duplicates.
        mag = np.abs(resid_corr)
        mag[S] = -np.inf
        S.append(int(np.argmax(mag)))
    assert len(set(S)) == N, "duplicate atom selected"
    best = solve_support(Gram, b, S, M)[1]
    for _ in range(max_rounds):
        improved = False
        for pos in range(N):
            for cand in range(D):
                if cand in S:
                    continue
                T = S.copy()
                T[pos] = cand
                e = solve_support(Gram, b, T, M)[1]
                if e > best + 1e-12:
                    best, S, improved = e, T, True
                    break
            if improved:
                break
        if not improved:
            break
    return sorted(S), best


# ------------------------------------------------------------ branch and bound
def bnb(Gram, b, yy, N, M, incumbent_S, incumbent_expl, max_nodes=400_000,
        time_limit=120.0, log=print):
    """Best-first branch-and-bound. Returns (support, error, certified gap)."""
    D = len(b)
    half = 0.5 * yy
    best_expl, best_S = incumbent_expl, list(incumbent_S)

    def node_bound(I):
        """(*) evaluated at a node. Returns (LB on error, residual corr, R)."""
        if I:
            c, expl = solve_support(Gram, b, I, M)
            R = yy - expl
            rc = b - Gram[:, I] @ c
        else:
            R, rc = yy, b.copy()
        R = max(R, 0.0)
        return R, rc

    heap = [(0.0, 0, [], set())]                 # (lb, tiebreak, I, excluded)
    counter = 1
    nodes = 0
    t0 = time.time()
    front = 0.0
    while heap:
        if nodes >= max_nodes or time.time() - t0 > time_limit:
            front = heap[0][0]
            break
        lb, _, I, O = heapq.heappop(heap)
        front = lb
        if lb >= half - 0.5 * best_expl - 1e-12:
            # everything remaining is worse than the incumbent
            heap = []
            front = lb
            break
        nodes += 1
        k = N - len(I)
        free = [j for j in range(D) if j not in O and j not in I]
        if k == 0:
            continue
        if len(free) < k:
            continue
        R, rc = node_bound(I)
        mag = np.abs(rc)
        mag_free = mag[free]
        order = np.argsort(-mag_free)
        sigma = float(mag_free[order[:k]].sum())
        num = R - M * sigma
        child_lb = (num * num) / (2 * R) if num > 0 and R > 0 else 0.0
        if child_lb >= half - 0.5 * best_expl - 1e-12:
            continue
        jstar = int(free[order[0]])
        # include j*
        I2 = I + [jstar]
        if len(I2) == N:
            c2, e2 = solve_support(Gram, b, I2, M)
            if e2 > best_expl:
                best_expl, best_S = e2, I2
        else:
            heapq.heappush(heap, (child_lb, counter, I2, set(O)))
            counter += 1
        # exclude j*
        O2 = set(O)
        O2.add(jstar)
        if D - len(O2) - len(I) >= k:
            heapq.heappush(heap, (child_lb, counter, list(I), O2))
            counter += 1
    err = half - 0.5 * best_expl
    if not heap and nodes < max_nodes:
        gap = 0.0                                   # tree exhausted: proven
        front = err
    else:
        gap = max(0.0, (err - front) / max(err, 1e-300))
    return sorted(best_S), err, gap, nodes, time.time() - t0, front


# ------------------------------------------------------------------ experiment
def run(name, n=48, N=6, kind="parabolic", Mmult=(2.0,), max_nodes=400_000,
        time_limit=120.0, log=print):
    X, Y = _grid(n)
    rng = np.random.default_rng(0)
    y = e2b.target(name, n, rng)
    yy = float(y @ y)
    half = 0.5 * yy
    sc = n / 32.0
    if kind == "parabolic":
        specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
                 (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
        th, G = e4.build_parabolic(n, specs, alpha=2.5)
    else:
        th, G, _, _ = e4.build_dict(n, max(1, int(4 * sc)),
                                    (2.0 * sc, 3.5 * sc, 6.0 * sc), (0.5, 1.0), 3)
    D = len(G)
    Gram, b = G @ G.T, G @ y
    coh = float(np.abs(Gram - np.eye(D)).max())
    pc = lambda e: 100.0 * e / half
    log(f"# target={name} n={n} dict={kind} D={D} coherence={coh:.4f} N={N}")
    for mm in Mmult:
        Sg, eg = greedy_then_swap(Gram, b, N, np.inf)
        c0, _ = solve_support(Gram, b, Sg, np.inf)
        M = mm * float(np.abs(c0).max())
        Sg, eg = greedy_then_swap(Gram, b, N, M)
        err_g = half - 0.5 * eg
        S, err, gap, nodes, t, front = bnb(Gram, b, yy, N, M, Sg, eg,
                                           max_nodes=max_nodes,
                                           time_limit=time_limit, log=log)
        status = "PROVEN OPTIMAL" if gap <= 1e-12 else f"gap <= {100*gap:.2f}%"
        log(f"  M={mm:.1f}x|c|max={M:8.3f}  greedy+swap {pc(err_g):8.4f}  "
            f"B&B {pc(err):8.4f}  improvement {100*(err_g-err)/err_g:+6.2f}%  "
            f"{status}  [{nodes} nodes, {t:.0f}s]")
    return dict(D=D, coh=coh)


def main(targets=("cartoon", "ascent", "face"), out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E8: exact l0 branch-and-bound, certified global optimum over a box.")
    for t in targets:
        for N in (4, 6, 8):
            log("")
            try:
                run(t, N=N, log=log)
            except Exception:
                import traceback
                log(f"# {t} N={N} FAILED:")
                for ln in traceback.format_exc().splitlines():
                    log("#   " + ln)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    tg = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else \
        ("cartoon", "ascent", "face")
    main(targets=tg, out=sys.argv[2] if len(sys.argv) > 2 else None)
