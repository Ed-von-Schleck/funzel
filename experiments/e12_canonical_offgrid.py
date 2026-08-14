"""E12 (U23): is the optimum's uniqueness a property of the problem, or of the grid?

E11 found the optimum sharply unique and stable -- one support in 2,511,496
within 1% of the best, unchanged by 30dB noise -- but measured it by enumerating
a GRID. Section 2.2 already established that discretisation imposes a minimum
separation, and that this separation is what makes sparsity convex-representable
at all. The same separation is a plausible cause of a unique argmin. If so,
canonicity is something bought by discretising rather than a property of the
continuous problem, and the encoder anyone actually wants is off-grid.

This decides between those two readings.

PART A -- the decisive one. At N=3, run many independent random restarts of a
full CONTINUOUS optimisation (Adam then L-BFGS, the combination E11 found
strongest) and ask whether they converge on the same answer. Two outcomes with
opposite meanings, and they are distinguished exactly as E11 distinguished them:
by whether the scattered solutions share an ERROR.
  * restarts agree, or disagree only in error -> a single continuous optimum
    that is merely hard to reach: canonicity is real, and the problem is
    optimisation;
  * restarts disagree at EQUAL error -> genuinely many continuous optima:
    canonicity fails off-grid and is an artefact of the grid.

PART B. The basin of the grid optimum. Start from it, displace by a known
radius, refine continuously, and measure whether the solution returns. A
canonical optimum should have a basin wide compared to the perturbations that
E11 showed leave it unmoved (30dB noise, a one-pixel shift).

PART C. The other direction the grid could be flattering: N. Uniqueness is
measured at N=3 only, and degeneracy should grow with atom count since more
atoms means more ways to trade them against each other. Enumerating N=2..5 on
one fixed dictionary gives the trend. D is held constant across N so the
comparison is clean, which forces D small enough that C(D,5) is affordable.

WHAT THIS CANNOT SHOW. N=3 for parts A and B, one image size, and a continuous
optimiser that is itself imperfect -- part A's negative branch would be
suggestive rather than conclusive, since "many optima" and "one optimum my
optimiser never finds" are separated only by the error test, which assumes the
restarts got close to their local optima.
"""

import sys
import time

import numpy as np

from e2_relaxation_gap import _grid, atoms, fit_fixed_support
import e2b_natural as e2b
import e4_exact_l0 as e4
import e5_dictionary_scaling as e5
import e11_canonical as e11


def setup(name, n=48):
    X, Y = _grid(n)
    y = e2b.target(name, n, np.random.default_rng(0))
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    return X, Y, y, th, G, 0.5 * float(y @ y)


def polish(th0, amp0, y, X, Y, n, adam=4000, lb=1200):
    """Adam then L-BFGS -- E11 found Adam stronger, L-BFGS sharper at the end."""
    lo, hi = e5.bounds(n)
    e_a, th_a = e11.adam_fit(th0, amp0, y, X, Y, n, adam, 0.01)
    G = atoms(np.clip(th_a, lo, hi), X, Y)[0]
    c = np.linalg.lstsq(G.T, y, rcond=None)[0]
    c2, th2, f = fit_fixed_support(c, np.clip(th_a, lo, hi), y, X, Y, 0.0,
                                   lo, hi, maxiter=lb)
    return float(f), th2


# ------------------------------------------------------------------ Part A
def part_a(name, n=48, N=3, n_restarts=40, log=print):
    X, Y, y, thg, G, half = setup(name, n)
    pc = lambda e: 100.0 * e / half
    # grid optimum, for reference
    Gram, b = G @ G.T, G @ y
    C, errs = e11.all_errors(G, y, N, Gram, b, half)
    S0 = C[int(np.argmin(errs))]
    e_grid = float(errs.min())

    res = []
    for s in range(n_restarts):
        g = np.random.default_rng(7000 + s)
        th0 = np.column_stack([
            g.uniform(0.05, 0.95, N), g.uniform(0.05, 0.95, N),
            np.log(g.uniform(2.0, 12.0, N)), np.zeros(N),
            np.log(g.uniform(2.0, 12.0, N))])
        A = atoms(th0, X, Y)[0]
        amp0 = np.linalg.lstsq(A.T, y, rcond=None)[0]
        res.append(polish(th0, amp0, y, X, Y, n))
    errs_r = np.array([r[0] for r in res])
    best = float(errs_r.min())
    ib = int(np.argmin(errs_r))

    log(f"  {name}: grid optimum {pc(e_grid):.4f}%, "
        f"best of {n_restarts} continuous restarts {pc(best):.4f}%")
    for tol in (0.001, 0.01, 0.05):
        near = np.flatnonzero(errs_r <= best * (1 + tol))
        # among restarts that reached this error, how far apart are the atoms?
        sp = [e11.atom_spread(res[ib][1], res[i][1], n) for i in near if i != ib]
        med = float(np.median(sp)) if sp else 0.0
        log(f"      within {100*tol:4.1f}% of the best restart: "
            f"{len(near):3d}/{n_restarts} restarts, "
            f"median atom spread among them {med:5.2f}px")
    # the discriminator: do the NON-converged restarts differ in error too?
    log(f"      error spread over all restarts: "
        f"{pc(errs_r.min()):.4f}% .. {pc(np.median(errs_r)):.4f}% "
        f"(median) .. {pc(errs_r.max()):.4f}%")
    d_grid = e11.atom_spread(thg[S0], res[ib][1], n)
    log(f"      best continuous solution sits {d_grid:.2f}px from the grid "
        f"optimum, and is {100*(e_grid-best)/e_grid:+.1f}% better in error")
    return dict(e_grid=e_grid, best=best, errs=errs_r)


# ------------------------------------------------------------------ Part B
def part_b(name, n=48, N=3, n_seeds=8, log=print):
    X, Y, y, thg, G, half = setup(name, n)
    pc = lambda e: 100.0 * e / half
    Gram, b = G @ G.T, G @ y
    C, errs = e11.all_errors(G, y, N, Gram, b, half)
    S0 = C[int(np.argmin(errs))]
    th_star = thg[S0]
    A = atoms(th_star, X, Y)[0]
    amp_star = np.linalg.lstsq(A.T, y, rcond=None)[0]
    e_ref, th_ref = polish(th_star, amp_star, y, X, Y, n)
    log(f"  {name}: grid optimum refined off-grid to {pc(e_ref):.4f}% "
        f"(moved {e11.atom_spread(th_star, th_ref, n):.2f}px)")
    for rad_px in (0.5, 1.0, 2.0, 4.0, 8.0):
        back, errs_b = [], []
        for s in range(n_seeds):
            g = np.random.default_rng(8000 + s)
            th0 = th_star.copy()
            th0[:, :2] += g.normal(0, rad_px / n, (N, 2))
            th0[:, 2] += g.normal(0, 0.1, N)
            th0[:, 4] += g.normal(0, 0.1, N)
            Ai = atoms(np.clip(th0, *e5.bounds(n)), X, Y)[0]
            a0 = np.linalg.lstsq(Ai.T, y, rcond=None)[0]
            e, thp = polish(th0, a0, y, X, Y, n)
            back.append(e11.atom_spread(th_ref, thp, n))
            errs_b.append(e)
        ret = sum(1 for d in back if d < 1.0)
        log(f"      displaced {rad_px:4.1f}px: {ret}/{n_seeds} returned within "
            f"1px, median return distance {np.median(back):5.2f}px, "
            f"median error {pc(float(np.median(errs_b))):.4f}%")


# ------------------------------------------------------------------ Part C
def part_c(name, n=48, log=print):
    X, Y, y, thg, G, half = setup(name, n)
    # one fixed, smaller dictionary so C(D,N) is affordable up to N=5 and the
    # comparison across N is not confounded by changing the dictionary
    keep = np.arange(0, len(G), max(1, len(G) // 90))[:90]
    Gs = G[keep]
    Gram, b = Gs @ Gs.T, Gs @ y
    log(f"  {name}: D={len(Gs)} held fixed across N")
    for N in (2, 3, 4, 5):
        C, errs = e11.all_errors(Gs, y, N, Gram, b, half)
        best = errs.min()
        counts = [int((errs <= best * (1 + t)).sum()) for t in (0.001, 0.01, 0.05)]
        log(f"      N={N}: {len(C):>10,} supports, best {100*best/half:7.4f}%,"
            f"  within 0.1%/1%/5% of best: {counts[0]:4d} / {counts[1]:5d} /"
            f" {counts[2]:6d}")


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E12 (U23): is the optimum's uniqueness the problem's, or the grid's?")
    log("")
    log("## A. Do independent continuous restarts converge on one answer?")
    for t in ("cartoon", "ascent"):
        part_a(t, log=log)
    log("")
    log("## B. How wide is the basin of the optimum?")
    for t in ("cartoon",):
        part_b(t, log=log)
    log("")
    log("## C. Does uniqueness degrade with N? (grid, D fixed)")
    for t in ("cartoon", "ascent"):
        part_c(t, log=log)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
