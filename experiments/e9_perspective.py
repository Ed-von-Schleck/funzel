"""E9: is the perspective relaxation strong enough to certify here?

E8 implemented exact l0 branch-and-bound and found the big-M node bound loose
by 6-9x at the tightest admissible box, so no search strategy could close a
gap. The remaining candidate, and the one the sparse-regression literature
actually uses, is the PERSPECTIVE relaxation. This measures whether it would
have worked, without building a second full solver: if the relaxation is weak
at the ROOT it is weak everywhere, and branch-and-bound is hopeless before the
first branch.

THE RELAXATION, derived rather than cited (both source PDFs are blocked by the
egress proxy, so nothing here rests on a summary). Write the cardinality-
constrained ridge problem as a mixed-integer program,

    min 0.5||y - Gc||^2 + lambda2 * sum_j c_j^2   s.t.  sum_j z_j <= N,
    z_j in {0,1},  c_j = 0 whenever z_j = 0,

and replace c_j^2 by its perspective c_j^2 / z_j, then relax z to [0,1]^D. For
fixed c the inner problem is a water-filling,

    Omega(c) = min { sum_j c_j^2 / z_j : sum_j z_j <= N, 0 <= z_j <= 1 },

solved by z_j = min(1, |c_j| / sqrt(mu)) with mu set so the budget binds. With
the largest s coordinates saturated at z=1,

    Omega(c) = sum_{saturated} c_j^2  +  (sum_{rest} |c_j|)^2 / (N - s).

Two properties make this the right object. Omega(c) >= ||c||^2 always, with
equality exactly when c has at most N non-zeros, so

    min_c 0.5||y - Gc||^2 + lambda2 * Omega(c)

is a genuine lower bound on the N-sparse ridge optimum and is TIGHT on every
feasible point. And it is convex, so the bound is computable exactly rather
than estimated.

WHY THE RIDGE IS NOT OPTIONAL. As lambda2 -> 0 the penalty vanishes and the
bound degenerates to zero: an unpenalised least-squares fit over the whole
dictionary. So the perspective relaxation can only certify a problem that
CARRIES a ridge, and the certified statement is about l0+ridge, not (P0). The
interesting quantity is therefore not "does it work" but "how much ridge does
it need, and does that much ridge still describe the problem we care about".
Both are measured below: the relaxation gap against the exhaustively computed
N-sparse ridge optimum, and the ridge term's size relative to the data term.

WHAT THIS CAN SHOW. The exact root relaxation gap as a function of lambda2, and
hence whether branch-and-bound on this formulation could close at these budgets.

WHAT THIS CANNOT SHOW. A root gap is necessary, not sufficient: a small root gap
does not prove the tree is small. And this is one dictionary, one image size,
N<=4, on-grid.
"""

import sys

import numpy as np
from scipy.optimize import minimize

from e2_relaxation_gap import _grid
import e2b_natural as e2b
import e4_exact_l0 as e4


def omega(c, N, eps=0.0):
    """Water-filling value and the optimal z, plus the gradient factor c/z."""
    a = np.sqrt(c * c + eps * eps) if eps > 0 else np.abs(c)
    idx = np.argsort(-a)
    s_sorted = a[idx]
    best = None
    csum = np.concatenate([[0.0], np.cumsum(s_sorted)])
    total = csum[-1]
    for s in range(0, min(N, len(a))):
        rest = total - csum[s]
        if N - s <= 0:
            break
        sqrt_mu = rest / (N - s)
        hi = s_sorted[s - 1] if s > 0 else np.inf
        lo = s_sorted[s] if s < len(a) else 0.0
        if sqrt_mu <= hi + 1e-12 and sqrt_mu >= lo - 1e-12:
            best = (s, sqrt_mu)
            break
    if best is None:
        best = (0, total / N if N > 0 else np.inf)
    s, sqrt_mu = best
    z = np.ones_like(a)
    if sqrt_mu > 0:
        z = np.minimum(1.0, a / sqrt_mu)
    z = np.maximum(z, 1e-12)
    val = float((c * c / z).sum())
    return val, z


def relax_value(G, y, N, lam2, warm=None, maxiter=3000, eps=1e-8):
    """min_c 0.5||y-Gc||^2 + lam2*Omega(c). Convex, but NOT smooth: for an
    unsaturated coordinate z_j = |c_j|/sqrt(mu), so c_j/z_j = sqrt(mu)*sign(c_j)
    and the gradient jumps at zero exactly as l1 does. Plain L-BFGS stalls there
    and returns a value ABOVE the true minimum, which is not a valid lower bound
    -- that is what produced impossible negative gaps in the first run.

    Two repairs, and the bound is only reported when both agree.
      * magnitudes are smoothed to sqrt(c^2+eps^2). Since that only ever
        increases the apparent magnitude, it increases z, hence DEcreases Omega
        and the minimum, so the smoothed value stays a valid lower bound.
      * the solve is warm-started at the exact N-sparse optimum, where
        Omega = ||c||^2 exactly, so the objective there equals the true optimum.
        Descent from that point can only go down, which guarantees the returned
        value never exceeds the optimum it is supposed to bound."""
    D = G.shape[0]

    def f(c):
        r = G.T @ c - y
        om, z = omega(c, N, eps=eps)
        val = 0.5 * float(r @ r) + lam2 * om
        grad = G @ r + 2.0 * lam2 * (c / z)
        return val, grad

    best = np.inf
    bx = None
    for c0 in ([np.zeros(D)] if warm is None else [warm, np.zeros(D)]):
        res = minimize(f, c0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-16,
                                "gtol": 1e-14})
        if res.fun < best:
            best, bx = float(res.fun), res.x
    return best, bx


def exact_ridge_l0(G, y, N, lam2, Gram, b, yy):
    """Exhaustive N-sparse optimum of the SAME ridge objective.

    Normal equations for a fixed support are (Gram_S + 2*lam2*I) c = b_S, so the
    objective value follows from the Gram matrix alone -- no pixel access."""
    D = G.shape[0]
    C = np.arange(D, dtype=np.int32)[:, None]
    for _ in range(N - 1):
        C = e4._extend(C, D)
    best = np.inf
    bestS = None
    eye = 2.0 * lam2 * np.eye(N)
    for i in range(0, len(C), 500_000):
        S = C[i:i + 500_000]
        bS = b[S]
        GS = Gram[S[:, :, None], S[:, None, :]] + eye
        sol = np.linalg.solve(GS, bS[..., None])[..., 0]
        # 0.5||y-Gc||^2 + lam2||c||^2 = 0.5 yy - c.b + 0.5 c'(Gram+2lam2 I)c
        obj = 0.5 * yy - (bS * sol).sum(1) + 0.5 * np.einsum(
            "ij,ijk,ik->i", sol, GS, sol)
        k = int(np.argmin(obj))
        if obj[k] < best:
            best, bestS = float(obj[k]), S[k].copy()
    return best, bestS


def run(name="cartoon", n=48, N=4, log=print):
    X, Y = _grid(n)
    y = e2b.target(name, n, np.random.default_rng(0))
    yy = float(y @ y)
    half = 0.5 * yy
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    D = len(G)
    Gram, b = G @ G.T, G @ y
    log(f"# target={name} n={n} D={D} N={N} ||y||^2={yy:.2f} "
        f"(0.5||y||^2={half:.2f})")
    log("")
    log(f"{'lambda2':>9} {'exact l0+ridge':>15} {'relaxation':>11} "
        f"{'root gap':>9} {'ridge/data':>11} {'l2 err':>9}")
    rows = []
    for lam2 in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        opt, S = exact_ridge_l0(G, y, N, lam2, Gram, b, yy)
        warm = np.zeros(D)
        GS0 = Gram[np.ix_(S, S)] + 2 * lam2 * np.eye(N)
        warm[S] = np.linalg.solve(GS0, b[S])
        rel, crel = relax_value(G, y, N, lam2, warm=warm)
        # C5: a relaxation cannot exceed what it relaxes. Warm-starting at the
        # exact optimum makes this structural, so a violation is a solver bug.
        assert rel <= opt + 1e-6 * abs(opt), (
            f"FAIL-C5: relaxation {rel:.6f} > exact optimum {opt:.6f}")
        # decompose the exact solution: how much of it is the ridge term?
        bS = b[S]
        GS = Gram[np.ix_(S, S)] + 2 * lam2 * np.eye(N)
        c = np.linalg.solve(GS, bS)
        data = 0.5 * yy - c @ bS + 0.5 * c @ Gram[np.ix_(S, S)] @ c
        ridge = lam2 * float(c @ c)
        gap = (opt - rel) / max(abs(opt), 1e-300)
        log(f"{lam2:9.4g} {opt:15.4f} {rel:11.4f} {100*gap:8.2f}% "
            f"{ridge/max(data,1e-12):11.2f} {100*data/half:8.4f}%")
        rows.append(dict(lam2=lam2, opt=opt, rel=rel, gap=gap,
                         ratio=ridge / max(data, 1e-12)))
    log("")
    log("  root gap 0% would mean branch-and-bound closes without branching;")
    log("  ridge/data is how large the regulariser is relative to the")
    log("  reconstruction error it is supposed to be a mild correction to.")
    log("  'l2 err' is the pure reconstruction error of the ridge solution,")
    log("  as % of 0.5||y||^2, comparable with every other table in the repo.")
    return rows


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E9: how strong is the perspective relaxation on the splatting problem?")
    for name in ("cartoon", "ascent"):
        log("")
        log("=" * 76)
        try:
            run(name, log=log)
        except Exception:
            import traceback
            log(f"# {name} FAILED:")
            for ln in traceback.format_exc().splitlines():
                log("#   " + ln)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else None)
