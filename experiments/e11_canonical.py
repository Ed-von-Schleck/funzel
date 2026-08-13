"""E11: is the optimum CANONICAL? And how does any of this compare to Adam?

Two questions, one experiment, because they turn out to be the same question.

WHY CANONICITY IS THE REAL TARGET. The motivation for wanting a global optimum
rather than a local one is not reconstruction quality -- section 10 already
shows the quality differences are modest. It is that "optimal" means
"canonical": the global minimiser is a FUNCTION OF THE INPUT, so it is
reproducible across runs and comparable across images, which is what makes a
representation analysable for patterns. A local optimum is an artifact of the
initialisation and the optimiser, and two of them cannot be meaningfully
compared.

That target needs two properties nobody here has tested, and reaching the
global optimum is not sufficient for either:

  A. UNIQUENESS. If a thousand distinct supports sit within a hair of the best,
     the argmin is decided by rounding, and "the" optimum is not a well-defined
     object even though a minimum value exists.
  B. STABILITY. A canonical representation must vary continuously with the
     input. If a one-pixel shift or 40dB of noise sends the optimal support
     somewhere unrelated, the map from image to atoms is reproducible for a
     fixed input and useless for comparing two inputs.

With a dictionary of coherence 0.985 both are in genuine doubt. Both are
measurable exactly at N=3 by enumerating every support, which is what parts A
and B do -- no optimiser is involved, so nothing here is confounded by solver
behaviour.

PART C is the comparison against standard practice. Everything in this
repository has been benchmarked against random-init plus a few hundred L-BFGS
iterations. The field -- GaussianImage included -- uses random init plus tens
of thousands of ADAM steps and no densification, which is a different regime,
not the same baseline at a different budget. A 2x2 of {random, greedy} x
{L-BFGS, Adam} at matched gradient evaluations settles whether careful
placement survives contact with the optimiser everyone actually uses.

Part C reports the cross-seed SPREAD of the solutions as well as their error,
because under the canonicity framing that is the more important number: a
method that reaches a slightly worse error but the SAME atoms every time is
more useful than one that reaches a better error somewhere different each run.

The Adam learning rate is swept rather than guessed. Three headline results in
this repository turned on a free parameter chosen once and never varied
(section 10.5's scale cap, 10.6's mapping constants, 10.7's blur schedule, the
last of which reversed outright when swept), and the learning rate is exactly
such a parameter.

WHAT THIS CANNOT SHOW. N=3 on a grid for parts A and B, because exhaustive
enumeration is what makes them exact; degeneracy and stability could differ at
N=10^3. Part C is off-grid but at 8-32 splats on 64^2 images, still far from
the regime of interest.
"""

import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from e2_relaxation_gap import _grid, atoms, loss_grad, NP_ATOM
import e2b_natural as e2b
import e4_exact_l0 as e4
import e5_dictionary_scaling as e5

FAIL = []


# --------------------------------------------------------------- distances
def atom_spread(thA, thB, n):
    """Median matched centre distance in pixels between two atom sets.

    Optimal assignment, so the N! labelling symmetry is quotiented out -- two
    solutions that differ only by ordering score 0, as they should."""
    if len(thA) == 0 or len(thB) == 0:
        return np.nan
    A, B = thA[:, :2] * n, thB[:, :2] * n
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    r, c = linear_sum_assignment(D)
    return float(np.median(D[r, c]))


# ------------------------------------------------------ A: near-degeneracy
def all_errors(G, y, N, Gram, b, half):
    """Error of EVERY N-subset. Exact; no optimiser involved."""
    D = G.shape[0]
    C = np.arange(D, dtype=np.int32)[:, None]
    for _ in range(N - 1):
        C = e4._extend(C, D)
    errs = np.empty(len(C))
    for i in range(0, len(C), 1_000_000):
        S = C[i:i + 1_000_000]
        bS = b[S]
        GS = Gram[S[:, :, None], S[:, None, :]] + 1e-10 * np.eye(N)
        sol = np.linalg.solve(GS, bS[..., None])[..., 0]
        errs[i:i + len(S)] = half - 0.5 * (bS * sol).sum(axis=1)
    return C, errs


def part_a(name, n=48, N=3, log=print):
    X, Y = _grid(n)
    y = e2b.target(name, n, np.random.default_rng(0))
    half = 0.5 * float(y @ y)
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    Gram, b = G @ G.T, G @ y
    C, errs = all_errors(G, y, N, Gram, b, half)
    order = np.argsort(errs)
    best = errs[order[0]]
    S0 = set(C[order[0]].tolist())
    log(f"  {name:8s} D={len(G)} supports={len(C):,}  best={100*best/half:.4f}%")
    for tol in (0.001, 0.01, 0.05):
        k = int((errs <= best * (1 + tol)).sum())
        # of those near-ties, how many share NO atom with the winner?
        cand = C[errs <= best * (1 + tol)]
        disjoint = sum(1 for s in cand if not (set(s.tolist()) & S0))
        log(f"      within {100*tol:4.1f}% of best: {k:7,d} supports"
            f"   ({100*k/len(C):.3f}% of all),  {disjoint:6,d} share NO atom "
            f"with the winner")
    # how far apart, geometrically, are the top solutions?
    tops = [th[C[order[i]]] for i in range(1, 6)]
    d = [atom_spread(th[C[order[0]]], t, n) for t in tops]
    log(f"      runners-up 2..6: matched centre distance from the winner = "
        f"{', '.join(f'{v:.1f}px' for v in d)}")
    return dict(best=best, half=half)


# ---------------------------------------------------------- B: stability
def part_b(name, n=48, N=3, log=print):
    X, Y = _grid(n)
    y0 = e2b.target(name, n, np.random.default_rng(0))
    half = 0.5 * float(y0 @ y0)
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    Gram = G @ G.T

    def opt(y):
        C, errs = all_errors(G, y, N, Gram, G @ y, 0.5 * float(y @ y))
        k = int(np.argmin(errs))
        return C[k]

    S0 = opt(y0)
    rng = np.random.default_rng(1)
    rows = []
    for label, y1 in [
        ("noise 40dB", y0 + rng.normal(0, np.sqrt((y0 @ y0) / len(y0) * 1e-4),
                                       len(y0))),
        ("noise 30dB", y0 + rng.normal(0, np.sqrt((y0 @ y0) / len(y0) * 1e-3),
                                       len(y0))),
        ("shift 1px", np.roll(y0.reshape(n, n), 1, axis=1).ravel()),
        ("scale x1.01", y0 * 1.01),
    ]:
        S1 = opt(y1)
        shared = len(set(S0.tolist()) & set(S1.tolist()))
        d = atom_spread(th[S0], th[S1], n)
        rows.append((label, shared, d))
        log(f"      {label:12s} shares {shared}/{N} atoms with the "
            f"unperturbed optimum, matched centre distance {d:5.2f}px")
    return rows


# ------------------------------------------------------------- C: Adam 2x2
def adam_fit(th0, amp0, y, X, Y, n, steps, lr):
    """Plain Adam on the same objective and gradient L-BFGS uses."""
    lo, hi = e5.bounds(n)
    K = len(amp0)
    z = np.concatenate([amp0, np.clip(th0, lo, hi).ravel()])
    zlo = np.concatenate([np.full(K, -np.inf), np.tile(lo, K)])
    zhi = np.concatenate([np.full(K, np.inf), np.tile(hi, K)])
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = np.inf
    for t in range(1, steps + 1):
        f, g = loss_grad(z, y, X, Y, 0.0)
        best = min(best, f)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        z = z - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        z = np.clip(z, zlo, zhi)
    f, _ = loss_grad(z, y, X, Y, 0.0)
    K = len(amp0)
    return min(best, f), z[K:].reshape(K, NP_ATOM)


def part_c(name, n=64, budgets=(8, 32), n_seeds=3, lbfgs_iter=900,
           adam_steps=6000, lrs=(0.01, 0.03), log=print):
    X, Y = _grid(n)
    y = e2b.target(name, n, np.random.default_rng(0))
    half = 0.5 * float(y @ y)
    pc = lambda e: 100.0 * e / half
    dicts = e5.make_dicts(n)
    G, S = dicts["gauss unstructured"]
    log(f"  {name}: L-BFGS {lbfgs_iter} iters (<=~{2*lbfgs_iter} gradient evals) "
        f"vs Adam {adam_steps} steps = {adam_steps} evals, lr swept over {lrs}."
        f" Adam is therefore given ~3x MORE gradient evaluations.")
    log(f"      {'budget':>6} {'init':>7} {'optimiser':>10} {'error':>9} "
        f"{'cross-seed spread':>18}")
    out = []
    for B in budgets:
        inits = {}
        idx, c, _, _ = e5.greedy_splats(G, S, y, B)
        inits["greedy"] = [e5.expand(S, idx, c)]   # deterministic: one run suffices
        rr = []
        for s in range(n_seeds):
            g = np.random.default_rng(500 + s)
            th0 = np.column_stack([
                g.uniform(0.05, 0.95, B), g.uniform(0.05, 0.95, B),
                np.log(g.uniform(2.0, 20.0, B)), np.zeros(B),
                np.log(g.uniform(2.0, 20.0, B))])
            A = atoms(th0, X, Y)[0]
            rr.append((th0, np.linalg.lstsq(A.T, y, rcond=None)[0]))
        inits["random"] = rr
        for iname, seeds in inits.items():
            for oname in ("L-BFGS", "Adam"):
                errs, ths = [], []
                for (th0, amp0) in seeds:
                    if oname == "L-BFGS":
                        e = e5.refine(th0, amp0, y, X, Y, n, maxiter=lbfgs_iter)
                        lo, hi = e5.bounds(n)
                        from e2_relaxation_gap import fit_fixed_support
                        _, thf, _ = fit_fixed_support(
                            amp0, np.clip(th0, lo, hi), y, X, Y, 0.0, lo, hi,
                            maxiter=lbfgs_iter)
                    else:
                        e, thf = min(
                            (adam_fit(th0, amp0, y, X, Y, n, adam_steps, lr)
                             for lr in lrs), key=lambda t: t[0])
                    errs.append(e)
                    ths.append(thf)
                spread = (np.nan if iname == "greedy" else
                          float(np.median([atom_spread(ths[i], ths[j], n)
                                           for i in range(len(ths))
                                           for j in range(i + 1, len(ths))])))
                log(f"      {B:6d} {iname:>7} {oname:>10} "
                    f"{pc(float(np.median(errs))):9.4f} "
                    f"{('n/a (deterministic)' if np.isnan(spread) else f'{spread:8.2f}px'):>18}")
                out.append(dict(B=B, init=iname, opt=oname,
                                err=float(np.median(errs)), spread=spread))
    return out


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E11: canonicity of the optimum, and a comparison against Adam.")
    log("")
    log("## A. Is the optimum unique? (exact, every support enumerated, N=3)")
    for t in ("cartoon", "ascent", "face"):
        part_a(t, log=log)
    log("")
    log("## B. Is it stable under small changes to the input?")
    for t in ("cartoon", "ascent"):
        log(f"  {t}:")
        part_b(t, log=log)
    log("")
    log("## C. random/greedy x L-BFGS/Adam, error AND cross-seed spread")
    for t in ("cartoon", "ascent"):
        part_c(t, log=log)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
