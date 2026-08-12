"""E2b: the (P-lambda) penalty on targets OUTSIDE the model class.

E2 used targets that were exact K-atom Gaussian mixtures, so the (P0) optimum was
exactly 0 and the relaxation gap was measurable exactly. Real images are not in
the model class: there is no true support, "recovery" has no referent, and only
approximation exists. This is the case that matters for encoding.

Two consequences for what can be claimed.

1. No known (P0) optimum. We use the strongest direct method available as the
   reference -- certificate-driven greedy at lambda=0 with full polish, which
   recovered exactly in E2 -- plus random restarts, taking the best. Since
   E_ref >= E_opt, the reported difference E_BL - E_ref is a LOWER bound on the
   true relaxation gap. It can show the penalty is large; it cannot show it is
   small. What it does measure exactly is the practical question: at matched
   atom count, does the certified convex route reconstruct as well as a strong
   direct optimizer?

2. No separation ratio r. The analogue is atom density, swept via the budget N.
   To place results on E2's axis we report an EMPIRICAL separation ratio:
   median nearest-neighbour centre distance divided by median atom width, over
   the fitted atoms.

Targets: two real photographs and one Donoho-Candes cartoon (piecewise-C2 with a
C2 edge), the last being the class for which section 3's approximation rates are
stated.
"""

import numpy as np
from scipy.optimize import minimize

from e2_relaxation_gap import (_grid, atoms, render, fit_fixed_support,
                               shape_bank, eta_sup, NP_ATOM)

BOUNDS_LO = np.array([0.0, 0.0, np.log(1.0), -50.0, np.log(1.0)])
BOUNDS_HI = np.array([1.0, 1.0, np.log(400.0), 50.0, np.log(400.0)])


# ----------------------------------------------------------------------- data
def target(name, n, rng):
    if name == "cartoon":
        X, Y = _grid(n)
        x, y = X.reshape(n, n), Y.reshape(n, n)
        smooth1 = 0.35 + 0.30 * np.cos(2.6 * x + 0.7) * np.cos(2.1 * y - 0.4)
        smooth2 = 0.85 - 0.40 * (x - 0.3) ** 2 - 0.25 * y
        # C2 boundary: a smooth closed curve
        t = np.arctan2(y - 0.5, x - 0.5)
        rad = 0.26 + 0.06 * np.cos(3 * t) + 0.03 * np.sin(2 * t)
        inside = ((x - 0.5) ** 2 + (y - 0.5) ** 2) < rad ** 2
        return np.where(inside, smooth2, smooth1).ravel()
    import scipy.datasets as sd
    img = sd.ascent().astype(float) if name == "ascent" else \
        sd.face(gray=True).astype(float)
    h, w = img.shape
    i0, j0 = (h - min(h, w)) // 2, (w - min(h, w)) // 2
    s = min(h, w)
    img = img[i0:i0 + s, j0:j0 + s]
    k = s // n
    img = img[:k * n, :k * n].reshape(n, k, n, k).mean(axis=(1, 3))  # box downsample
    img -= img.min()
    return (img / img.max()).ravel()


# ------------------------------------------------------------------ reference
def greedy_p0(y, n, X, Y, bank, N, polish_every=4):
    """Certificate-driven greedy at lambda=0: the strongest (P0) proxy we have."""
    c = np.zeros(0)
    th = np.zeros((0, NP_ATOM))
    for k in range(N):
        resid = y - (render(c, th, X, Y) if c.size else 0.0)
        _, arg = eta_sup(resid.reshape(n, n), bank, n, X, Y, n_refine=4, nm_iter=150)
        c = np.concatenate([c, [0.0]])
        th = np.vstack([th, arg])
        it = 700 if (k + 1) % polish_every == 0 or k == N - 1 else 200
        c, th, f = fit_fixed_support(c, th, y, X, Y, 0.0, BOUNDS_LO, BOUNDS_HI,
                                     maxiter=it)
    return c, th, f


def restart_p0(y, X, Y, N, rng, n_restarts):
    best = (np.inf, None, None)
    for _ in range(n_restarts):
        th0 = np.column_stack([
            rng.uniform(0.05, 0.95, N), rng.uniform(0.05, 0.95, N),
            np.log(rng.uniform(3.0, 40.0, N)), rng.normal(0, 0.5, N),
            np.log(rng.uniform(3.0, 40.0, N))])
        c0 = rng.normal(0, 0.3, N)
        c, th, f = fit_fixed_support(c0, th0, y, X, Y, 0.0, BOUNDS_LO, BOUNDS_HI,
                                     maxiter=900)
        if f < best[0]:
            best = (f, c, th)
    return best


# ------------------------------------------------------------------- blasso
def blasso(y, n, X, Y, lam, bank, max_atoms, inner_iter=400):
    c = np.zeros(0)
    th = np.zeros((0, NP_ATOM))
    for _ in range(max_atoms):
        resid = y - (render(c, th, X, Y) if c.size else 0.0)
        sup, arg = eta_sup(resid.reshape(n, n), bank, n, X, Y, n_refine=4, nm_iter=150)
        if sup <= lam * 1.001:
            break
        c = np.concatenate([c, [0.0]])
        th = np.vstack([th, arg])
        c, th, _ = fit_fixed_support(c, th, y, X, Y, lam, BOUNDS_LO, BOUNDS_HI,
                                     maxiter=inner_iter)
        keep = np.abs(c) > 1e-4 * max(1.0, np.abs(c).max())
        c, th = c[keep], th[keep]
        if c.size == 0:
            break
    resid = y - (render(c, th, X, Y) if c.size else 0.0)
    sup, _ = eta_sup(resid.reshape(n, n), bank, n, X, Y, n_refine=4, nm_iter=150)
    p = resid * min(1.0, lam / max(sup, 1e-300))
    gap = (0.5 * resid @ resid + lam * np.abs(c).sum()) - (-0.5 * p @ p + p @ y)
    return c, th, gap


def sep_ratio(th, n):
    """Empirical separation: median NN centre distance / median atom width, in px."""
    if th.shape[0] < 2:
        return np.nan
    P = th[:, :2] * n
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    width = np.exp(-0.5 * (th[:, 2] + th[:, 4])) * n     # geometric-mean sigma, px
    return np.median(nn) / np.median(width)


# ---------------------------------------------------------------- experiment
def run(name="ascent", n=64, budgets=(8, 16, 32), n_restarts=2, seed=0):
    X, Y = _grid(n)
    rng = np.random.default_rng(seed)
    y = target(name, n, rng)
    half = 0.5 * y @ y
    bank = shape_bank(np.log(np.array([1.5, 2.5, 4.0, 6.5, 10.0, 16.0]) / n),
                      [0.35, 0.55, 0.8, 1.0], np.linspace(0, np.pi, 6, endpoint=False))
    print(f"# target={name} n={n} 0.5||y||^2={half:.3f}")
    print(f"{'N':>4} {'E_ref':>8} {'E_BLraw':>9} {'E_BLdb':>8} {'E_BLpol':>8} "
          f"{'penalty':>8} {'certgap':>8} {'N_BL':>5} {'r_emp':>6} {'r_ref':>7}",
          flush=True)
    rows = []
    for N in budgets:
        cg_, thg, f_g = greedy_p0(y, n, X, Y, bank, N)
        f_r, c_r, th_r = restart_p0(y, X, Y, N, rng, n_restarts)
        E_ref = min(f_g, f_r)

        sup0, _ = eta_sup(y.reshape(n, n), bank, n, X, Y, n_refine=4, nm_iter=150)
        lo, hi = 1e-4 * sup0, sup0
        pick = None
        for _ in range(6):
            lam = np.sqrt(lo * hi)
            c, th, gap = blasso(y, n, X, Y, lam, bank, max_atoms=int(2.5 * N) + 4)
            if pick is None or abs(c.size - N) < abs(pick[0].size - N):
                pick = (c, th, gap)
            if c.size > N:
                lo = lam
            elif c.size < N:
                hi = lam
            else:
                pick = (c, th, gap); break
        c, th, gap = pick
        r_bl = y - render(c, th, X, Y)
        E_bl = 0.5 * r_bl @ r_bl
        G = atoms(th, X, Y)[0]
        c_db = np.linalg.lstsq(G.T, y, rcond=None)[0]
        E_db = 0.5 * np.sum((y - c_db @ G) ** 2)
        c_pol, th_pol, E_pol = fit_fixed_support(c_db, th, y, X, Y, 0.0,
                                                 BOUNDS_LO, BOUNDS_HI, maxiter=900)
        # A polished BLASSO solution is itself a feasible N-atom point, so it
        # upper-bounds the (P0) optimum and belongs in the reference. Excluding it
        # would understate the penalty.
        E_ref = min(E_ref, E_pol)
        pc = lambda e: 100 * e / half
        print(f"{N:4d} {pc(E_ref):8.3f} {pc(E_bl):9.3f} {pc(E_db):8.3f} {pc(E_pol):8.3f} "
              f"{pc(E_db - E_ref):8.3f} {pc(gap):8.4f} {c.size:5d} "
              f"{sep_ratio(th, n):6.2f} {sep_ratio(thg, n):7.2f}", flush=True)
        rows.append(dict(N=N, E_ref=E_ref, E_bl=E_bl, E_db=E_db, E_pol=E_pol,
                         gap=gap, N_bl=int(c.size), r_emp=sep_ratio(th, n),
                         r_emp_ref=sep_ratio(thg, n), half=half))
    return rows


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "ascent")
