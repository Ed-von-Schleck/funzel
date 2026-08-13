"""E2: the (P-lambda)/(P0) relaxation gap as a function of atom overlap.

NOTE: a negative certgap is not a bug in the duality bound, it is evidence that
the grid+refine supremum underestimated max|eta|, leaving p dual-infeasible.
It is printed raw rather than clamped, as a diagnostic on the certificate.

Targets are sums of K anisotropic 2D Gaussians on a fixed centre grid; the only
swept variable is the width u, so the separation ratio r = (centre spacing)/u
runs from well-separated to densely overlapping.

At each r we compute
  (P0)       best-of-restarts fit with exactly K atoms, L-BFGS on all parameters
  (P-lambda) BLASSO via sliding Frank-Wolfe, lambda bisected to support size K
and report reconstruction error for both, plus a certified duality gap for the
BLASSO solve.

The measured gap E(P-lambda) - E(best-found P0) is a LOWER bound on the true
relaxation gap, since best-found >= true optimum. It can show the gap is large;
it cannot show it is small.

Atoms live on a truncated parameter space (positions in the image, log-scales
bounded), so the certificate refers to (P-lambda) restricted to that set.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

# ----------------------------------------------------------------- atom model
# theta = (cx, cy, u, v, w):  Sigma^{-1} = M^T M,  M = [[e^u, v], [0, e^w]]
NP_ATOM = 5


def _grid(n):
    ax = (np.arange(n) + 0.5) / n
    Y, X = np.meshgrid(ax, ax, indexing="ij")
    return X.ravel(), Y.ravel()


def atoms(theta, X, Y):
    """theta (K,5) -> (K, P) atom images."""
    theta = np.atleast_2d(theta)
    dx = X[None, :] - theta[:, 0:1]
    dy = Y[None, :] - theta[:, 1:2]
    eu, v, ew = np.exp(theta[:, 2:3]), theta[:, 3:4], np.exp(theta[:, 4:5])
    # M d = (eu*dx + v*dy, ew*dy);  q = 0.5 |M d|^2
    m0 = eu * dx + v * dy
    m1 = ew * dy
    # UNIT-NORM dictionary. ||phi_theta||^2 = n^2 * pi * exp(-(u+w)) analytically,
    # so the theta-dependent part of the norm is exp(-(u+w)/2); the constant
    # factor is global and folds into lambda. Without this the l1 penalty is not
    # commensurate across scales: a wide atom has a large norm and so buys more
    # inner product per unit of penalty, and argmax|eta| drifts to broad atoms
    # over empty regions. Classical BLASSO never sees this because its dictionary
    # is translation-only with a fixed kernel, hence of constant norm.
    z = np.exp(0.5 * (theta[:, 2:3] + theta[:, 4:5]))
    return z * np.exp(-0.5 * (m0 * m0 + m1 * m1)), m0, m1, dx, dy


def render(c, theta, X, Y):
    G = atoms(theta, X, Y)[0]
    return c @ G


def loss_grad(z, y, X, Y, lam, eps=1e-6):
    """z = [c (K,), theta (K,5)] flattened. Smoothed-l1 BLASSO objective."""
    K = z.size // (1 + NP_ATOM)
    c = z[:K]
    theta = z[K:].reshape(K, NP_ATOM)
    G, m0, m1, dx, dy = atoms(theta, X, Y)
    r = c @ G - y
    f = 0.5 * r @ r
    gc = G @ r
    # dphi/dparam = -phi * dq/dparam
    cg = c[:, None] * G                      # (K,P)
    dq_dcx = -(m0 * np.exp(theta[:, 2:3]))
    dq_dcy = -(m0 * theta[:, 3:4] + m1 * np.exp(theta[:, 4:5]))
    dq_du = m0 * dx * np.exp(theta[:, 2:3])
    dq_dv = m0 * dy
    dq_dw = m1 * dy * np.exp(theta[:, 4:5])
    # normalisation contributes +1/2 on the two log-scale components
    gth = np.stack([-(cg * d) @ r for d in (dq_dcx, dq_dcy,
                                            dq_du - 0.5, dq_dv, dq_dw - 0.5)], axis=1)
    if lam > 0:
        s = np.sqrt(c * c + eps * eps)
        f += lam * s.sum()
        gc = gc + lam * c / s
    return f, np.concatenate([gc, gth.ravel()])


def fit_fixed_support(c0, th0, y, X, Y, lam, bounds_lo, bounds_hi, maxiter=400):
    K = c0.size
    z0 = np.concatenate([c0, th0.ravel()])
    lo = np.concatenate([np.full(K, -np.inf), np.tile(bounds_lo, K)])
    hi = np.concatenate([np.full(K, np.inf), np.tile(bounds_hi, K)])
    res = minimize(loss_grad, z0, args=(y, X, Y, lam), jac=True, method="L-BFGS-B",
                   bounds=list(zip(lo, hi)), options={"maxiter": maxiter})
    K = res.x.size // (1 + NP_ATOM)
    return res.x[:K], res.x[K:].reshape(K, NP_ATOM), res.fun


# ------------------------------------------------------- certificate / FW step
def shape_bank(log_s, aspects, rots):
    """Candidate (u,v,w) shapes for the grid search over Sigma^{-1}."""
    out = []
    for ls in log_s:
        for a in aspects:
            for t in rots:
                s1, s2 = np.exp(ls), np.exp(ls) * a
                R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                Sig = R @ np.diag([s1 ** 2, s2 ** 2]) @ R.T
                A = np.linalg.inv(Sig)
                M = np.linalg.cholesky(A).T          # upper triangular
                out.append((np.log(M[0, 0]), M[0, 1], np.log(M[1, 1])))
    return np.array(out)


def eta_sup(resid_img, bank, n, X, Y, n_refine=8, nm_iter=200):
    """max_theta |<phi_theta, resid>| over truncated Theta, grid + local refine."""
    best = []
    for (u, v, w) in bank:
        th = np.array([[0.5, 0.5, u, v, w]])
        tmpl = atoms(th, X, Y)[0].reshape(n, n)
        # keep the kernel small: crop to where it matters
        corr = fftconvolve(resid_img, tmpl[::-1, ::-1], mode="same")
        k = np.argmax(np.abs(corr))
        iy, ix = divmod(k, n)
        best.append((abs(corr[iy, ix]), (ix + 0.5) / n, (iy + 0.5) / n, u, v, w))
    best.sort(key=lambda t: -t[0])
    rv = resid_img.ravel()

    def negcorr(p):
        G = atoms(p[None, :], X, Y)[0][0]
        return -abs(G @ rv)

    top = best[0][0]
    arg = np.array(best[0][1:])
    for cand in best[:n_refine]:
        p0 = np.array(cand[1:])
        r = minimize(negcorr, p0, method="Nelder-Mead",
                     options={"maxiter": nm_iter, "xatol": 1e-4, "fatol": 1e-8})
        if -r.fun > top:
            top, arg = -r.fun, r.x
    return top, arg


def blasso_fw(y, n, X, Y, lam, bank, bounds_lo, bounds_hi, max_atoms=25, tol=1e-3):
    """Sliding Frank-Wolfe. Returns (c, theta, certified duality gap)."""
    c = np.zeros(0)
    th = np.zeros((0, NP_ATOM))
    for _ in range(max_atoms):
        recon = render(c, th, X, Y) if c.size else np.zeros_like(y)
        resid = y - recon
        sup, arg = eta_sup(resid.reshape(n, n), bank, n, X, Y)
        if sup <= lam * (1 + tol):
            break
        c = np.concatenate([c, [0.0]])
        th = np.vstack([th, arg])
        c, th, _ = fit_fixed_support(c, th, y, X, Y, lam, bounds_lo, bounds_hi,
                                    maxiter=500)
        keep = np.abs(c) > 1e-4 * max(1.0, np.abs(c).max())
        c, th = c[keep], th[keep]
        if c.size == 0:
            break
    # certified gap: p = residual, rescaled to be dual feasible
    recon = render(c, th, X, Y) if c.size else np.zeros_like(y)
    resid = y - recon
    sup, _ = eta_sup(resid.reshape(n, n), bank, n, X, Y)
    p = resid * min(1.0, lam / max(sup, 1e-300))
    primal = 0.5 * resid @ resid + lam * np.abs(c).sum()
    dual = -0.5 * p @ p + p @ y
    return c, th, primal - dual, primal


# ------------------------------------------------------------------ experiment
def run(n=96, K=9, u_px=6.0, ratios=(6.0, 4.0, 3.0, 2.0, 1.5, 1.0),
        n_restarts=6, seed=0):
    """u_px is held FIXED so pixel resolution is constant across the sweep;
    the centre spacing is r*u, so r is the only thing that varies."""
    X, Y = _grid(n)
    rng = np.random.default_rng(seed)
    u = u_px / n

    bounds_lo = np.array([0.0, 0.0, np.log(1.0), -50.0, np.log(1.0)])
    bounds_hi = np.array([1.0, 1.0, np.log(200.0), 50.0, np.log(200.0)])

    print("all errors as % of 0.5*||y||^2;  (P0) optimum is exactly 0 by construction")
    print(f"{'r':>5} {'sep_px':>7} {'BLraw':>8} {'BLdebi':>8} {'BLpolish':>9} "
          f"{'P0rstrt':>9} {'certgap':>9} {'K_BL':>5} {'absE_db':>10} {'halfyy':>10}",
          flush=True)
    rows = []
    for r in ratios:
        spacing = r * u
        g = int(np.ceil(np.sqrt(K)))
        off = (np.arange(g) - (g - 1) / 2) * spacing
        cxs, cys = np.meshgrid(0.5 + off, 0.5 + off, indexing="ij")
        centres = np.stack([cxs.ravel(), cys.ravel()], 1)[:K]
        # The centre lattice grows with r, so a large separation can push the
        # ground-truth atoms OUTSIDE the image, leaving a near-empty target on
        # which every method scores ~0 and "exact recovery" is vacuous. This was
        # unchecked, and section 9.1's three zero rows sit exactly where it is
        # most likely to have happened. Fail loudly rather than report a zero.
        if centres.min() < 0.0 or centres.max() > 1.0:
            raise ValueError(
                f"r={r}: ground-truth centres span [{centres.min():.3f},"
                f"{centres.max():.3f}] outside the unit square. Need "
                f"(ceil(sqrt(K))-1)*r*u_px <= n; here K={K}, u_px={u_px}, n={n}."
                " Any error reported for this row would be measured on a target"
                " whose atoms are off-image.")
        # ground truth: mild anisotropy + random orientation, unit amplitudes
        th_gt = []
        for (cx, cy) in centres:
            t = rng.uniform(0, np.pi)
            s1, s2 = u, u * 0.65
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            A = np.linalg.inv(R @ np.diag([s1 ** 2, s2 ** 2]) @ R.T)
            M = np.linalg.cholesky(A).T
            th_gt.append([cx, cy, np.log(M[0, 0]), M[0, 1], np.log(M[1, 1])])
        th_gt = np.array(th_gt)
        c_gt = rng.uniform(0.6, 1.4, K)
        y = render(c_gt, th_gt, X, Y)
        ynorm = y @ y

        # ---- (P0): best of restarts at exactly K atoms (difficulty diagnostic only)
        best = np.inf
        for _ in range(n_restarts):
            th0 = np.column_stack([
                rng.uniform(0.15, 0.85, K), rng.uniform(0.15, 0.85, K),
                np.log(rng.uniform(0.5, 2.0, K) / u), rng.normal(0, 0.5, K),
                np.log(rng.uniform(0.5, 2.0, K) / u)])
            c0 = rng.normal(0, 0.5, K)
            _, _, f = fit_fixed_support(c0, th0, y, X, Y, 0.0, bounds_lo, bounds_hi)
            best = min(best, f)
        E_p0 = best

        # ---- (P-lambda): bisect lambda to support size K
        bank = shape_bank(np.log(u * np.array([0.4, 0.6, 0.85, 1.2, 1.7, 2.4])),
                          [0.4, 0.6, 0.8, 1.0], np.linspace(0, np.pi, 6, endpoint=False))
        sup0, _ = eta_sup(y.reshape(n, n), bank, n, X, Y)
        lo, hi = 1e-4 * sup0, sup0
        chosen = None
        for _ in range(6):
            lam = np.sqrt(lo * hi)
            c, th, cg, primal = blasso_fw(y, n, X, Y, lam, bank, bounds_lo, bounds_hi)
            if chosen is None or abs(c.size - K) < abs(chosen[0].size - K):
                chosen = (c, th, cg, primal, lam)
            if c.size > K:
                lo = lam
            elif c.size < K:
                hi = lam
            else:
                chosen = (c, th, cg, primal, lam)
                break
        c, th, cg, primal, lam = chosen
        r_bl = y - render(c, th, X, Y)
        E_bl = 0.5 * r_bl @ r_bl

        # debias: refit amplitudes on the BLASSO support, lambda = 0
        G = atoms(th, X, Y)[0]
        c_db = np.linalg.lstsq(G.T, y, rcond=None)[0]
        r_db = y - c_db @ G
        E_db = 0.5 * r_db @ r_db
        # polish: BLASSO support as an initializer, full local refit
        if c_db.size:
            _, _, E_pol = fit_fixed_support(c_db, th, y, X, Y, 0.0,
                                            bounds_lo, bounds_hi)
        else:
            # An empty BLASSO support crashed fit_fixed_support with an
            # unhelpful unpacking error from scipy. Report it as the trivial
            # all-zero fit instead, which is what an empty support means.
            E_pol = 0.5 * float(y @ y)

        # The target IS a sum of K Gaussians, so (P0) at N=K attains error exactly
        # 0 at the ground truth. The relaxation gap is therefore E_BL itself --
        # exact, not a bound. E_p0 is kept only as a difficulty diagnostic.
        # NB: 0.5*||y||^2 is NOT constant across r -- overlapping same-sign atoms
        # sum constructively, so it inflates as r falls. Percentages are therefore
        # not comparable across rows on their own; absolute E and the energy are
        # printed so the trend can be renormalized against a fixed reference.
        pc = lambda e: 100 * e / (0.5 * ynorm)
        print(f"{r:5.2f} {spacing*n:6.1f} {pc(E_bl):8.2f} {pc(E_db):8.2f} {pc(E_pol):9.3f} "
              f"{pc(E_p0):9.2f} {pc(cg):9.4f} {c.size:5d} {E_db:10.3f} {0.5*ynorm:10.2f}",
              flush=True)
        rows.append(dict(r=r, sep_px=spacing * n, E_p0=E_p0, E_bl=E_bl, E_db=E_db,
                         E_pol=E_pol, certgap=cg, K_bl=int(c.size), ynorm=ynorm))
    return rows


if __name__ == "__main__":
    run()
