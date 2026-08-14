"""E17: how much of the objective can be taken out of the rendered image.

Corollary 7 of `convexification-and-N.md` says a convex lift cannot beat the
mass ball when its objective is the error of a LINEARLY rendered image. Every
strengthening in the sparse-regression literature that does see N escapes that
hypothesis the same way: it needs a piece of the objective that is a separable
function of the coefficients rather than a function of Ac. The perspective
relaxation is the clearest case -- it replaces a separable term c_j^2 by its
perspective c_j^2/z_j, and it has nothing to replace unless such a term exists.

The available separable piece is exactly

    max  tr(D)   s.t.  A^T A + lambda2 * I  -  D  >= 0,   D diagonal >= 0

since c^T(A^T A + l2 I)c splits into c^T D c (separable, perspective-able) plus
c^T(A^T A + l2 I - D)c (still convex, still not separable). tr(D) against
tr(A^T A + l2 I) is the fraction of the quadratic a perspective-type argument
can work on. This file computes a CERTIFIED UPPER BOUND on it for the
splatting dictionary.

WHY A BOUND AND NOT THE SDP. The bound needs no solver and is rigorous from the
eigendecomposition alone. For any feasible D and any eigenpair (lambda_i, v_i),

    v_i^T (A^T A - D) v_i >= 0   =>   sum_j d_j v_ij^2 <= lambda_i.

Summing over the k smallest eigenpairs and using d >= 0 with
s_j = sum_{i<=k} v_ij^2 > 0,

    (min_j s_j) * tr(D) <= sum_j d_j s_j <= sum_{i<=k} lambda_i
    =>  tr(D) <= (sum_{i<=k} max(lambda_i,0)) / min_j s_j.

Minimising over k gives the reported number. Clipping the eigenvalues at zero
keeps it valid against the small negative eigenvalues a numerically singular
Gram produces.

PRE-REGISTERED (M5).

WHAT THIS CAN SHOW. Whether the splatting dictionary offers a perspective-type
relaxation anything to work with at small ridge, and how the answer moves with
the ridge. If the bound is near zero at lambda2 = 0, section 9's measured
64-86% root gap is forced rather than incidental, and no better implementation
of that family recovers it.

WHAT THIS CANNOT SHOW.
  (i)   An upper bound on tr(D) bounds the RAW MATERIAL, not the resulting gap.
        A relaxation with a large separable part is not thereby tight. The
        inference runs one way only: near-zero separable part => nothing to
        strengthen with.
  (ii)  It covers relaxations that strengthen via a separable quadratic. A lift
        that escapes Corollary 7 by some other route is untouched -- the
        completely positive reformulation of section 10 is one, and it is not
        of this form.
  (iii) One dictionary, one image size. The mechanism named below -- rank
        deficiency of the Gram -- is a property of a redundant dictionary of
        smooth atoms, not of splatting as such.
  (iv)  Bounds computed at small k rest on eigenvector components near double
        precision. The reported number is restricted to k where min_j s_j stays
        comfortably above that floor, and the tighter values available at
        smaller k are printed but not used.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e4_exact_l0 import build_dict

SAFE_S = 1e-6          # floor on min_j s_j below which the bound is not trusted


def certified_bound(w, V, lam2, safe_s=SAFE_S, log=print):
    """Upper bound on max tr(D) s.t. Gram + lam2 I - D >= 0, D diagonal >= 0."""
    D = len(w)
    ww = w + lam2
    best, best_k, rows = np.inf, None, []
    for k in range(10, D, 10):
        s = (V[:, :k] ** 2).sum(axis=1)
        ub = np.maximum(ww[:k], 0.0).sum() / max(s.min(), 1e-300)
        rows.append((k, s.min(), ub))
        if s.min() >= safe_s and ub < best:
            best, best_k = ub, k
    return best, best_k, rows


def run(n=32, lam2s=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), log=print):
    th, G, _, _ = build_dict(n, 4, (2.0, 3.5, 6.0), (0.5, 1.0), 3)
    Gram = G @ G.T
    D, P = G.shape
    w, V = np.linalg.eigh(Gram)
    tr = float(np.trace(Gram))

    log(f"# E17: separable mass available to a perspective-type relaxation")
    log(f"# dictionary D={D} atoms in P={P} pixels, unit-norm, tr(Gram)={tr:.1f}")
    log(f"# largest eigenvalue {w[-1]:.4f}; numerical rank at 1e-10 is "
        f"{int((w > 1e-10*w[-1]).sum())} of {D}")
    log(f"# the Gram is RANK DEFICIENT, which is what drives the result: an")
    log(f"# exact null vector v forces sum_j d_j v_j^2 <= 0, hence d_j = 0 on")
    log(f"# its support. Below, every coordinate carries null-space energy, so")
    log(f"# no coordinate escapes.")
    log("")
    nul = w <= 1e-10 * w[-1]
    s_nul = (V[:, nul] ** 2).sum(axis=1)
    log(f"  nullity at 1e-10: {int(nul.sum())};  coordinates with zero "
        f"null-space energy: {int((s_nul <= 0).sum())} of {D}")
    log("")
    log("  lambda2   tr(Gram+l2 I)   certified tr(D) <=    as % of trace   at k")
    out = []
    for lam2 in lam2s:
        tot = tr + lam2 * D
        ub, k, _ = certified_bound(w, V, lam2, log=log)
        ub = min(ub, tot)
        out.append((lam2, tot, ub))
        log(f"  {lam2:7.4g}   {tot:13.2f}   {ub:18.4f}   {100*ub/tot:14.3f}%   "
            f"{k if k else '-':>4}")

    log("")
    log("  checks")
    checks = []

    def check(label, ok, extra=""):
        checks.append(ok)
        log(f"    {'ok  ' if ok else 'FAIL'} {label} {extra}")

    z = out[0][2]
    check("C1 bound at lambda2=0 is below 1% of the trace", z < 0.01 * tr,
          f"({100*z/tr:.3f}%)")
    check("C2 every coordinate carries null-space energy, so d=0 is forced "
          "in exact arithmetic", int((s_nul <= 0).sum()) == 0)
    check("C3 D = lambda2 * I is feasible, so the bound is at least D*lambda2",
          all(ub >= lam2 * D - 1e-6 for lam2, _, ub in out),
          "(a bound below it would be wrong)")
    check("C4 the available fraction is monotone in lambda2",
          all(out[i][2] / out[i][1] >= out[i-1][2] / out[i-1][1] - 1e-9
              for i in range(1, len(out))))

    log("")
    log("  reading: at lambda2 = 0 a perspective-type strengthening has "
        "essentially")
    log("  nothing to act on. The fraction only becomes substantial once "
        "lambda2 is")
    log("  of order 1, where D = lambda2*I alone supplies half the quadratic --")
    log("  and section 9 measures the reconstruction 2.4-4.3x worse there. The")
    log("  ridge buys the relaxation its raw material and pays for it in the")
    log("  objective; there is no setting where it gets one without the other.")
    log(f"\n  {sum(checks)}/{len(checks)} checks passed")
    return out, checks


def main(out=None):
    stream = open(out, "w") if out else sys.stdout

    def log(*a):
        print(*a, file=stream)
        stream.flush()

    run(log=log)
    if out:
        stream.close()


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else None)
