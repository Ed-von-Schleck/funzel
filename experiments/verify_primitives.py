"""Verification suite for the primitives every result in this repo depends on.

MOTIVATION. Three bugs surfaced this session, and every one was caught the same
way: it produced a value that was impossible rather than merely wrong (a
negative duality gap, a local search scoring worse than its own starting point,
a relaxation exceeding the quantity it relaxes). That detector has narrow
coverage by construction -- it fires only when a defect pushes a number across a
known-forbidden boundary, and says nothing about a defect that moves 2.41 to
2.83. So the three caught bugs imply a detection RATE, not a total, and the
undetected population is precisely the bugs that produce plausible numbers.

Note also that all three were in code paths that HAPPENED to have such a check.
The paths without one carry no evidence in either direction.

This suite replaces impossibility with two stronger kinds of check, applied to
the primitives that every experiment routes through:

  * REDUNDANT COMPUTATION -- recompute the same quantity by an independent path
    (scipy's multivariate normal, itertools.combinations, brute-force search,
    numerical differentiation) and demand agreement;
  * ANALYTIC IDENTITY -- assert a property that must hold exactly (Parseval,
    Omega(c) = ||c||^2 on sparse c, budget accounting).

The gradient test is the load-bearing one. E5, E6 and E7 all conclude things
about off-grid REFINEMENT, so all three rest on fit_fixed_support finding good
optima. A subtly wrong analytic gradient would silently degrade every one of
those results without ever producing an impossible value -- and section 7.1.1
records that exactly this class of bug once invalidated two complete sweeps and
three interpretations here.

Run: python3 verify_primitives.py
"""

import itertools
import sys

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "**FAIL**"
    print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


# --------------------------------------------------------------------------
def t_atoms_against_scipy():
    """atoms() recomputed through scipy's multivariate normal density.

    Independent path: the code evaluates z*exp(-0.5|Md|^2) in a fused
    expression; this goes through Sigma = (M'M)^{-1} and a library pdf."""
    from scipy.stats import multivariate_normal
    from e2_relaxation_gap import _grid, atoms
    n = 24
    X, Y = _grid(n)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(12):
        t = np.array([rng.uniform(.2, .8), rng.uniform(.2, .8),
                      rng.uniform(1.0, 3.0), rng.uniform(-3, 3),
                      rng.uniform(1.0, 3.0)])
        got = atoms(t[None, :], X, Y)[0][0]
        M = np.array([[np.exp(t[2]), t[3]], [0.0, np.exp(t[4])]])
        Sig = np.linalg.inv(M.T @ M)
        pts = np.stack([X - t[0], Y - t[1]], axis=1)
        pdf = multivariate_normal(mean=[0, 0], cov=Sig).pdf(pts)
        # pdf = (2pi)^-1 |Sig|^-1/2 exp(-0.5 d'Sig^-1 d); recover the exponential
        expo = pdf * 2 * np.pi * np.sqrt(np.linalg.det(Sig))
        want = np.exp(0.5 * (t[2] + t[4])) * expo
        worst = max(worst, float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)))
    return check("atoms() vs scipy multivariate_normal", worst < 1e-9,
                 f"max rel err {worst:.2e}")


def t_gradient_finite_difference():
    """loss_grad's analytic gradient against central finite differences.

    THE load-bearing test: every refinement result in E5/E6/E7 depends on this
    gradient being right, and a wrong one degrades results silently."""
    from e2_relaxation_gap import _grid, loss_grad
    n = 16
    X, Y = _grid(n)
    rng = np.random.default_rng(1)
    worst = 0.0
    for lam in (0.0, 0.37):
        for _ in range(6):
            K = 3
            th = np.column_stack([
                rng.uniform(.25, .75, K), rng.uniform(.25, .75, K),
                rng.uniform(1.2, 2.4, K), rng.uniform(-1, 1, K),
                rng.uniform(1.2, 2.4, K)])
            c = rng.normal(0, 1, K)
            y = rng.normal(0, 1, n * n)
            z = np.concatenate([c, th.ravel()])
            _, g = loss_grad(z, y, X, Y, lam)
            gn = np.zeros_like(z)
            for i in range(z.size):
                h = 1e-6 * max(1.0, abs(z[i]))
                zp, zm = z.copy(), z.copy()
                zp[i] += h
                zm[i] -= h
                gn[i] = (loss_grad(zp, y, X, Y, lam)[0]
                         - loss_grad(zm, y, X, Y, lam)[0]) / (2 * h)
            rel = np.abs(g - gn) / np.maximum(np.abs(gn), 1e-6)
            worst = max(worst, float(rel.max()))
    return check("loss_grad analytic gradient vs finite differences",
                 worst < 2e-4, f"max rel err {worst:.2e}")


def t_norm_constant_interior():
    """The 'unit-norm' convention must hold where the atom is not clipped."""
    from e2_relaxation_gap import _grid, atoms
    n = 64
    X, Y = _grid(n)
    rng = np.random.default_rng(2)
    norms = []
    for _ in range(60):
        sig = rng.uniform(2.0, 5.0) / n            # comfortably interior
        t = np.array([0.5, 0.5, np.log(1 / sig), 0.0, np.log(1 / sig)])
        a = atoms(t[None, :], X, Y)[0][0]
        norms.append(float(np.linalg.norm(a)))
    spread = max(norms) / min(norms)
    return check("atom norm constant for interior atoms", spread < 1.01,
                 f"spread {spread:.5f}x, analytic {n*np.sqrt(np.pi):.1f} vs "
                 f"measured {np.median(norms):.1f}")


def t_extend_combinations():
    """_extend() must enumerate exactly C(D,k) strictly increasing tuples."""
    import e4_exact_l0 as e4
    from math import comb
    ok = True
    detail = ""
    for D, k in ((9, 3), (12, 4), (7, 2)):
        C = np.arange(D, dtype=np.int32)[:, None]
        for _ in range(k - 1):
            C = e4._extend(C, D)
        got = {tuple(r) for r in C.tolist()}
        want = set(itertools.combinations(range(D), k))
        if got != want or len(C) != comb(D, k):
            ok = False
            detail = f"D={D} k={k}: got {len(C)}, want {comb(D,k)}"
    return check("_extend() vs itertools.combinations", ok, detail)


def t_explained_energy():
    """exact_l0's Gram-domain explained energy vs an explicit lstsq fit."""
    rng = np.random.default_rng(3)
    P, D = 60, 14
    G = rng.normal(0, 1, (D, P))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    y = rng.normal(0, 1, P)
    Gram, b = G @ G.T, G @ y
    worst = 0.0
    for S in itertools.combinations(range(D), 3):
        S = list(S)
        c = np.linalg.lstsq(G[S].T, y, rcond=None)[0]
        direct = float(y @ y - np.sum((y - c @ G[S]) ** 2))
        gram_way = float(b[S] @ np.linalg.solve(Gram[np.ix_(S, S)], b[S]))
        worst = max(worst, abs(direct - gram_way) / max(abs(direct), 1e-12))
    return check("Gram-domain explained energy vs explicit lstsq",
                 worst < 1e-8, f"max rel err {worst:.2e}")


def t_sup_corr_bruteforce():
    """sup_corr must not UNDERSTATE the supremum -- the one error that would
    invalidate section 10.2 rather than merely weaken it. Compared against a
    dense brute-force scan over the same Theta."""
    from e2_relaxation_gap import _grid, atoms, NP_ATOM
    import e3_absolute_bound as e3
    n = 32
    X, Y = _grid(n)
    lo, hi = e3.theta_box(n)
    bank = e3.dense_bank(n)
    rng = np.random.default_rng(4)
    q = rng.normal(0, 1, n * n)
    s, _ = e3.sup_corr(q, n, X, Y, bank, lo, hi, rng, n_refine=8, n_random=8000)
    brute = 0.0
    for i in range(0, 40000, 4000):
        th = np.column_stack([rng.uniform(lo[k], hi[k], 4000)
                              for k in range(NP_ATOM)])
        brute = max(brute, float(np.abs(atoms(th, X, Y)[0] @ q).max()))
    return check("sup_corr() >= dense brute-force scan", s >= brute * (1 - 1e-9),
                 f"sup_corr {s:.4f} vs brute {brute:.4f}")


def t_ray_bound_closed_form():
    """ray_bound's closed form vs numerical maximisation over the ray.

    Tested in the direction that carries meaning. The closed form is the exact
    maximum, so it must never fall BELOW a sampled value -- that is what a real
    defect would look like, and it is checked exactly. The reverse gap is
    bounded by the grid's own sampling error, 0.5*||q||^2*(h/2)^2, not by a
    relative tolerance: when the optimum sits near zero (t* can be ~1e-3 against
    a 1e-4 step) a relative comparison is dominated by the grid and says nothing
    about the code. An earlier version of this test used one and 'failed' on a
    2.4e-8 absolute discrepancy that was below its own resolution."""
    import e3_absolute_bound as e3
    rng = np.random.default_rng(5)
    h = 1e-4
    below, worst_excess = 0, 0.0
    for _ in range(200):
        P = 40
        y = rng.normal(0, 1, P)
        q = rng.normal(0, 1, P)
        s = abs(rng.normal(0, 1)) * 0.3
        M = abs(rng.normal(0, 1))
        closed = e3.ray_bound(q, y, s, M)
        ts = np.arange(0.0, 20.0, h)
        nmax = max(float((ts * (q @ y) - 0.5 * ts ** 2 * (q @ q)
                          - M * ts * s).max()), 0.0)
        if closed < nmax - 1e-12:
            below += 1
        grid_err = 0.5 * float(q @ q) * (h / 2) ** 2
        worst_excess = max(worst_excess, (closed - nmax) - grid_err)
    return check("ray_bound closed form vs numerical max over t",
                 below == 0 and worst_excess <= 1e-9,
                 f"{below} values below sampled max; excess over grid error "
                 f"{worst_excess:.2e}")


def t_omega_bruteforce():
    """omega()'s water-filling vs brute-force minimisation over z."""
    from scipy.optimize import minimize
    import e9_perspective as e9
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(20):
        D, N = 7, 3
        c = rng.normal(0, 1, D)
        val, _ = e9.omega(c, N)
        f = lambda z: float((c * c / np.maximum(z, 1e-12)).sum())
        best = np.inf
        for _ in range(25):
            z0 = rng.uniform(0.05, 1.0, D)
            z0 *= min(1.0, N / z0.sum())
            r = minimize(f, z0, method="SLSQP",
                         bounds=[(1e-6, 1.0)] * D,
                         constraints=[{"type": "ineq",
                                       "fun": lambda z: N - z.sum()}])
            best = min(best, float(r.fun))
        worst = max(worst, abs(val - best) / max(abs(best), 1e-12))
    return check("omega() water-filling vs numerical minimisation",
                 worst < 1e-4, f"max rel err {worst:.2e}")


def t_splat_accounting():
    """E5's budget must equal the sum of the selected atoms' splat costs, and
    expand() must reproduce each dictionary atom from its splats."""
    from e2_relaxation_gap import _grid, atoms
    import e5_dictionary_scaling as e5
    n = 32
    X, Y = _grid(n)
    dicts = e5.make_dicts(n)
    ok, detail = True, ""
    for key, (G, S) in dicts.items():
        worst = 0.0
        for j in range(0, len(G), max(1, len(G) // 20)):
            th, w = S[j]
            rebuilt = w @ atoms(th, X, Y)[0]
            worst = max(worst, float(np.abs(rebuilt - G[j]).max()))
        if worst > 1e-8:
            ok, detail = False, f"{key}: atom rebuild err {worst:.2e}"
        y = atoms(np.array([[0.5, 0.5, 3.0, 0.0, 3.0]]), X, Y)[0][0]
        idx, c, _, used = e5.greedy_splats(G, S, y, 8)
        if used != sum(len(S[j][1]) for j in idx):
            ok, detail = False, f"{key}: budget accounting {used}"
    return check("E5 splat accounting and atom reconstruction", ok, detail)


def t_parseval():
    import e7_frequency_continuation as e7
    import e2b_natural as e2b
    n = 64
    y = e2b.target("cartoon", n, np.random.default_rng(0))
    lhs = float(y @ y)
    rhs = float((np.abs(np.fft.fft2(y.reshape(n, n))) ** 2).sum()) / n ** 2
    ok1 = abs(lhs - rhs) / lhs < 1e-12
    y0 = e7.lowpass(y, n, 0.0)
    ok2 = float(np.abs(y0 - y).max()) < 1e-10
    return check("Parseval identity, and lowpass(sigma=0) is the identity",
                 ok1 and ok2, f"rel {abs(lhs-rhs)/lhs:.2e}")


def main():
    tests = [t_atoms_against_scipy, t_gradient_finite_difference,
             t_norm_constant_interior, t_extend_combinations,
             t_explained_energy, t_sup_corr_bruteforce,
             t_ray_bound_closed_form, t_omega_bruteforce,
             t_splat_accounting, t_parseval]
    print("Verifying primitives by redundant computation and analytic identity.")
    print("(Impossibility checks are deliberately NOT used here -- they are what")
    print(" already ran, and their coverage is the thing in question.)\n")
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            check(t.__name__, False, f"raised {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print(f"All {len(tests)} primitive checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
