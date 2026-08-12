"""E3: an absolute lower bound on the (P0) optimum, and greedy's distance from it.

Section 10 is a RELATIVE result: greedy beats BLASSO. U14 observes that nothing
bounds greedy's distance from the (P0) optimum itself, so "use matching pursuit"
could be a statement about the best of two mediocre options. This experiment
supplies the missing absolute bound.

U14 as written in section 11 proposes "a fine exhaustive grid over (mu,Sigma)
with least-squares amplitudes gives a genuine lower bound on the (P0) optimum".
That method cannot work. An exhaustive grid search returns a feasible point, so
it UPPER bounds E_opt; it can show greedy is beatable but never bounds how far
greedy is from optimal. It is also infeasible: a fine grid has ~1e5 atoms, so
choosing 8 of them is ~1e36 subsets. Leg A replaces it; Leg B salvages the
grid idea in the one place it is both exact and affordable.

--------------------------------------------------------------------- Leg A
A certified lower bound. For ANY vector p and ANY measure m, two elementary
inequalities hold with no assumptions whatsoever:

    0.5||r||^2 >= <p,r> - 0.5||p||^2            (since 0.5||r-p||^2 >= 0)
    <Phi* p, m> <= sup_Theta |<phi_theta,p>| * |m|(Theta)      (Hoelder)

Chaining them with r = y - Phi m, for any m with |m|(Theta) <= M:

    0.5||y - Phi m||^2  >=  <p,y> - 0.5||p||^2 - M * s(p),   s(p) = sup|<phi,p>|

Along a ray p = t*q this is a concave quadratic in t, maximised in closed form:

    B(q; M) = (<q,y> - M*s(q))_+^2 / (2||q||^2)                          (*)

so each candidate direction q costs exactly three scalars, one of which is the
eta_sup computation already in the codebase. Taking the max of (*) over any
family of directions is still a valid lower bound, and adding directions can
only improve it. No convexity of the feasible set, no separation condition, no
non-degeneracy, and -- crucially -- NO REQUIREMENT THAT ANY SOLVER CONVERGED.
Solver quality affects only tightness. This experiment therefore does not
depend on U5, unlike U7 and U8.

WHAT THIS CAN SHOW. E_greedy - L(M_g) upper bounds how far greedy is from
optimal among encodings of total mass <= M_g. If small, greedy is certified
near-optimal and U14 is answered in the affirmative at that budget.

WHAT THIS CANNOT SHOW. L is a LOWER bound, so a large gap L << E_greedy is
inconclusive -- it may be bound looseness, not greedy suboptimality. Four
sources of looseness, all pre-registered before the run (M5):
  (i)   the bound constrains total mass, not atom count, so it bounds a
        RELAXATION of (P0):  L <= E_relaxed(M) <= E_Natom(M). Its slack is at
        least the relaxation gap of section 9 (1.0-4.8% out-of-model). This
        experiment cannot beat that and does not claim to.
  (ii)  writing q for greedy's own residual gives B/E_greedy = (1-rho)^2 with
        rho = M_g*s(q)/(2*E_greedy). rho grows with N, so the bound is EXPECTED
        to be informative at small N and to go vacuous somewhere above it. The
        budget grid is dense at the low end for that reason, and the N at which
        it dies is itself the measurement.
  (iii) it is conditional on the mass budget M. A better N-atom solution using
        MORE mass is not excluded. The mass sweep is what makes this visible;
        the conclusion must always be quoted with its M.
  (iv)  it holds over the truncated Theta of theta_box(), and "mass" means
        sum|c_i| in the dictionary normalisation of e2_relaxation_gap, which is
        unit-norm ANALYTICALLY but only approximately so on a pixel grid.
        norm_spread() measures that approximation rather than assuming it.

VALIDITY HINGES ENTIRELY ON s(q) NOT BEING UNDERESTIMATED. An underestimated
supremum inflates L and could falsely certify greedy as near-optimal -- the same
failure mode that produced the negative duality gap in section 9.4, which is
where this risk was first identified. Five independent guards:
  - positions are searched EXACTLY (FFT correlation evaluates every pixel offset
    for each shape), so only the 3-d shape space is ever sampled;
  - a dense shape bank, an independent uniform random probe of Theta, and
    Nelder-Mead refinement projected back into Theta;
  - a safety sweep reporting L with s inflated by 0%, 5%, 10% and 25%;
  - C1: L(M_g) <= E_greedy must hold, since greedy's own solution has mass M_g
    and so lies inside the feasible set the bound covers. Checked every row.
  - C2: on an in-model target, ground truth attains error exactly 0 at mass
    M_true, so L(M_true) <= 0 must hold. Amplitudes are near-equal by design,
    which makes this test nearly TIGHT rather than slack: for q=y the required
    inequality ||y||^2 <= M_true*s(y) becomes an equality when the atoms are
    orthogonal with equal amplitudes, so any underestimate of s pushes it
    positive immediately.
  A violation of C1 or C2 is proof of an underestimated supremum and is
  reported as a failure, never absorbed.

--------------------------------------------------------------------- Leg B
An upper bound, attacking the same question from the other side. On a discrete
dictionary small enough to enumerate, best-2 is available in closed form for
every pair, so greedy's myopia is measurable EXACTLY at N=2:

    best-2 exhaustive   vs   best-2 greedy (fix argmax, then best partner)

Both use the identical dictionary, so the difference isolates the cost of
committing to the first atom by matched filter, free of any confound from the
continuous refinement.

Two traps, both handled. (a) A pair of near-duplicate atoms with large opposing
amplitudes reduces error arbitrarily as their coherence -> 1; that is real
arithmetic but a useless encoding, and would let "exhaustive" win by artifact.
(b) Such a pair is also exactly what a mass budget forbids. So the exhaustive
search is reported three ways: unconstrained, coherence-capped, and
mass-matched to greedy's own pair -- the last being the honest comparison and
the one that connects to Leg A. Exact, but restricted to N=2 and to a grid; it
says nothing about N=8, and is reported separately for that reason.

--------------------------------------------------------------------- N=1
A third, smaller measurement that falls out for free. Greedy places its first
atom at argmax |<phi,y>|, but the true N=1 optimum is at argmax
<phi,y>^2/||phi||^2. These coincide only if the dictionary is EXACTLY unit-norm.
Section 7.1.1 fixed the analytic norm; on a pixel grid the discrete norm still
varies, so the two argmaxes can differ. Searching both objectives measures what
that residual mismatch costs, at the one budget where the (P0) optimum is
computable without any combinatorics.
"""

import sys
import time
import traceback

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from e2_relaxation_gap import (_grid, atoms, render, fit_fixed_support,
                               shape_bank, NP_ATOM)
import e2b_natural as e2b


# --------------------------------------------------------------- parameter space
def theta_box(n):
    """The truncated Theta, identical for every method compared here.

    Scales are held to resolved widths: e^u, e^w in [4, n] is roughly
    sigma in [1px, n/4px]. Section 7.1.1 fixed the WIDE end of the scale range by
    normalising the dictionary; the NARROW end needs a bound too, because the
    analytic norm n^2*pi*exp(-(u+w)) that the normalisation divides out is the
    CONTINUUM norm. Below about one pixel the discrete atom stops resolving and
    its true grid norm exceeds the analytic one, so a sub-pixel spike would be
    over-weighted -- section 7.1.1's failure at the opposite end of the scale.

    Note the direction of the residual error: an over-weighted atom can only
    INFLATE s(q), which weakens the bound. It cannot invalidate it.
    """
    lo = np.array([0.0, 0.0, np.log(4.0), -0.25 * n, np.log(4.0)])
    hi = np.array([1.0, 1.0, np.log(float(n)), 0.25 * n, np.log(float(n))])
    return lo, hi


def dense_bank(n):
    """Shape bank for the supremum search. Denser than E2/E2b's: here an
    incomplete bank threatens VALIDITY, whereas in E2 it only slowed the solver."""
    sig_px = np.array([1.2, 1.7, 2.4, 3.4, 4.8, 6.8, 9.6, 13.6])
    sig_px = sig_px[sig_px <= n / 4.0]
    return shape_bank(np.log(sig_px / n),
                      [0.3, 0.45, 0.6, 0.8, 1.0],
                      np.linspace(0, np.pi, 8, endpoint=False))


def norm_spread(n, X, Y, lo, hi, rng, n_samples=4000):
    """Measured discrete L2 norm of the 'unit-norm' atoms over Theta.

    The dictionary divides out the analytic continuum norm; on a pixel grid the
    true norm is not exactly constant. Reported so that 'total mass' has a
    stated meaning rather than an assumed one. Also reports the fraction of
    Theta whose minor axis is sub-pixel, since that is where the convention is
    worst."""
    th = np.column_stack([rng.uniform(lo[k], hi[k], n_samples) for k in range(NP_ATOM)])
    nr, minor = [], []
    for i in range(0, n_samples, 500):
        blk = th[i:i + 500]
        G = atoms(blk, X, Y)[0]
        nr.append(np.sqrt((G * G).sum(axis=1)))
        for t in blk:
            M = np.array([[np.exp(t[2]), t[3]], [0.0, np.exp(t[4])]])
            minor.append(1.0 / (np.linalg.svd(M, compute_uv=False)[0]) * n)
    nr, minor = np.concatenate(nr), np.array(minor)
    # split by width so the CAUSE is identified rather than guessed: boundary
    # truncation of wide atoms and sub-pixel narrow atoms break the analytic
    # norm at opposite ends, and they need different fixes.
    wid = np.exp(-0.5 * (th[:, 2] + th[:, 4])) * n           # geometric-mean sigma, px
    qlo, qhi = np.quantile(wid, [0.25, 0.75])
    return (nr.min(), np.median(nr), nr.max(), float(np.mean(minor < 1.0)),
            float(np.median(nr[wid <= qlo])), float(np.median(nr[wid >= qhi])))


def norm_attribution(n=64, n_samples=6000, seed=1, log=print):
    """Why the 'unit-norm' dictionary is not unit-norm on a pixel grid.

    norm_spread() reports THAT the analytic convention is off; this reports
    WHY, which needs an attribution rather than an eyeballed quartile split.
    Candidate causes -- sub-pixel minor axis, wide-atom boundary clipping, and
    shear -- are separated by rank correlation and by conditioning on each."""
    from scipy.stats import spearmanr
    X, Y = _grid(n)
    lo, hi = theta_box(n)
    rng = np.random.default_rng(seed)
    th = np.column_stack([rng.uniform(lo[k], hi[k], n_samples)
                          for k in range(NP_ATOM)])
    nr = []
    for i in range(0, n_samples, 500):
        G = atoms(th[i:i + 500], X, Y)[0]
        nr.append(np.sqrt((G * G).sum(axis=1)))
    nr = np.concatenate(nr)
    ana = n * np.sqrt(np.pi)                       # the analytic convention
    sv = np.array([np.linalg.svd(np.array([[np.exp(t[2]), t[3]],
                                           [0.0, np.exp(t[4])]]),
                                 compute_uv=False) for t in th])
    minor, major = n / sv[:, 0], n / sv[:, 1]      # px
    edge = np.minimum.reduce([th[:, 0], 1 - th[:, 0],
                              th[:, 1], 1 - th[:, 1]]) * n
    deficit = nr / ana

    log(f"# analytic norm {ana:.1f}; measured min {nr.min():.1f} "
        f"med {np.median(nr):.1f} max {nr.max():.1f} "
        f"(spread {nr.max()/nr.min():.2f}x over {n_samples} draws)")
    log("# rank correlation of norm/analytic with each candidate cause:")
    for nm, v in (("minor axis px", minor), ("major axis px", major),
                  ("edge distance px", edge), ("|v| shear", np.abs(th[:, 3]))):
        log(f"#   {nm:18s} rho = {spearmanr(deficit, v).statistic:+.3f}")
    far, res = edge > 2 * major, minor >= 1.0
    for nm, m in (("minor axis >= 1px", res),
                  ("support clears the edge (edge > 2*major)", far),
                  ("both", far & res)):
        log(f"#   conditioning on {nm}: {int(m.sum())} atoms, "
            f"spread {nr[m].max()/nr[m].min():.2f}x")
    return dict(spread=float(nr.max() / nr.min()),
                spread_interior=float(nr[far].max() / nr[far].min()))


# ------------------------------------------------------- supremum of |<phi,q>|
def sup_corr(q, n, X, Y, bank, lo, hi, rng, n_refine=24, nm_iter=400,
             n_random=24000, chunk=1500, normalize=False):
    """s(q) = sup_{theta in Theta} |<phi_theta,q>|, searched hard.

    Underestimating this INVALIDATES the bound, so the search is deliberately
    more thorough than E2's solver-side eta_sup: positions are exact via FFT,
    the shape bank is dense, an independent random probe covers what the
    structured bank might miss, and refinement is projected back into Theta so
    the argmax stays admissible.

    normalize=True instead maximises |<phi,q>|/||phi||, whose square is the
    error reduction of a single least-squares atom. Used only for the N=1 exact
    optimum, never for the bound."""
    q_img = q.reshape(n, n)

    def score(th_blk):
        G = atoms(th_blk, X, Y)[0]
        c = np.abs(G @ q)
        if normalize:
            c = c / np.maximum(np.sqrt((G * G).sum(axis=1)), 1e-300)
        return c

    # exact over all pixel positions, for each shape in the bank
    cands = []
    for (u, v, w) in bank:
        tmpl = atoms(np.array([[0.5, 0.5, u, v, w]]), X, Y)[0].reshape(n, n)
        corr = np.abs(fftconvolve(q_img, tmpl[::-1, ::-1], mode="same"))
        if normalize:
            corr = corr / max(np.linalg.norm(tmpl), 1e-300)
        k = int(np.argmax(corr))
        iy, ix = divmod(k, n)
        cands.append((float(corr[iy, ix]), (ix + 0.5) / n, (iy + 0.5) / n, u, v, w))

    # independent uniform probe of Theta, in case the structured bank has a hole
    th = np.column_stack([rng.uniform(lo[k], hi[k], n_random) for k in range(NP_ATOM)])
    for i in range(0, n_random, chunk):
        blk = th[i:i + chunk]
        c = score(blk)
        j = int(np.argmax(c))
        cands.append((float(c[j]), *blk[j]))

    cands.sort(key=lambda t: -t[0])
    s, arg = cands[0][0], np.array(cands[0][1:])

    # local refinement, projected into Theta
    def neg(p):
        return -float(score(np.clip(p, lo, hi)[None, :])[0])

    for cand in cands[:n_refine]:
        r = minimize(neg, np.array(cand[1:]), method="Nelder-Mead",
                     options={"maxiter": nm_iter, "xatol": 1e-5, "fatol": 1e-10})
        if -r.fun > s:
            s, arg = -r.fun, np.clip(r.x, lo, hi)
    return float(s), arg


# ------------------------------------------------------------------------- greedy
def greedy(y, n, X, Y, bank, N, lo, hi, rng, budgets=()):
    """Certificate-driven greedy at lambda=0 -- the section 10 method, run on the
    Theta of theta_box(). Structurally identical to e2b.greedy_p0 (same
    polish_every=4 schedule, so this IS section 10's greedy), reimplemented only
    because that function bakes its bounds in as module constants and the bound
    requires every method to share one Theta. Budgets that will be REPORTED get
    a full polish regardless of the schedule, so no reported row is degraded by
    where it happens to fall in the cycle."""
    c = np.zeros(0)
    th = np.zeros((0, NP_ATOM))
    out = []
    for k in range(N):
        resid = y - (render(c, th, X, Y) if c.size else 0.0)
        _, arg = sup_corr(resid, n, X, Y, bank, lo, hi, rng,
                          n_refine=6, nm_iter=200, n_random=6000)
        c = np.concatenate([c, [0.0]])
        th = np.vstack([th, arg])
        full = (k + 1) % 4 == 0 or k == N - 1 or (k + 1) in budgets
        c, th, f = fit_fixed_support(c, th, y, X, Y, 0.0, lo, hi,
                                     maxiter=700 if full else 200)
        out.append((k + 1, float(f), float(np.abs(c).sum()),
                    y - render(c, th, X, Y)))
    return out


# ---------------------------------------------------------------------- the bound
def ray_bound(q, y, s, M, qq=None, qy=None):
    """max_t [ <tq,y> - 0.5||tq||^2 - M*s(tq) ], in closed form. See (*)."""
    qq = float(q @ q) if qq is None else qq
    qy = float(q @ y) if qy is None else qy
    num = qy - M * s
    return (num * num) / (2.0 * qq) if num > 0 and qq > 0 else 0.0


def best_bound(dirs, y, M, infl=1.0):
    """Max of (*) over a family of directions. Valid for any family whatsoever."""
    return max(ray_bound(q, y, s * infl, M, qq, qy) for (q, s, qq, qy) in dirs)


def ascend(dirs, y, M, n_, X, Y, bank, lo, hi, rng, n_steps=18):
    """Supergradient ascent on p for a fixed mass budget M, appending every
    iterate to the shared direction pool.

        B(p) = <p,y> - 0.5||p||^2 - M*s(p)

    is CONCAVE in p (linear, minus a convex quadratic, minus M times a
    supremum of linear forms), so this is a genuine ascent rather than a
    heuristic. A supergradient is  y - p - M*sigma*phi_theta*  with theta* the
    argmax attaining s(p) and sigma the sign of <phi_theta*,p>; the step
    p <- (1-g)p + g(y - M*sigma*phi) with g = 2/(k+2) is the Frank-Wolfe
    averaging rule, which is also what SFW uses.

    Every iterate is appended with its OWN measured s, so validity never
    depends on this converging -- a bad ascent just fails to improve the max."""
    q0 = max(dirs, key=lambda d: ray_bound(d[0], y, d[1], M, d[2], d[3]))
    t = (q0[3] - M * q0[1]) / q0[2] if q0[2] > 0 else 0.0
    p = (t if t > 0 else 1.0 / max(np.linalg.norm(q0[0]), 1e-12)) * q0[0]
    sup = lambda v: sup_corr(v, n_, X, Y, bank, lo, hi, rng,
                             n_refine=6, nm_iter=250, n_random=8000)
    s, th = sup(p)                     # one supremum per iterate, not two
    for k in range(n_steps):
        phi = atoms(th[None, :], X, Y)[0][0]
        sigma = np.sign(phi @ p) or 1.0
        g = 2.0 / (k + 2.0)
        p = (1.0 - g) * p + g * (y - M * sigma * phi)
        if p @ p <= 0:
            break
        s, th = sup(p)
        # only the DIRECTION matters -- ray_bound already optimises the scale --
        # so each iterate earns its place by pointing somewhere new.
        dirs.append((p.copy(), s, float(p @ p), float(p @ y)))
    return dirs


# ------------------------------------------------------------------------- targets
def make_target(name, n, rng):
    """Returns (y, meta); meta carries ground truth for the in-model control."""
    if name == "inmodel":
        X, Y = _grid(n)
        u = 3.0 / n                              # 3px atoms
        centres = [(0.25, 0.25), (0.75, 0.27), (0.27, 0.75), (0.75, 0.75)]
        th = np.array([[cx, cy, np.log(1.0 / u), 0.0, np.log(1.0 / (0.65 * u))]
                       for (cx, cy) in centres])
        # near-equal amplitudes make check C2 nearly tight rather than slack
        c = np.array([1.0, 0.98, 1.02, 1.0])
        lo, hi = theta_box(n)
        assert np.all(th >= lo) and np.all(th <= hi), "ground truth outside Theta"
        return render(c, th, X, Y), dict(K=4, M_true=float(np.abs(c).sum()))
    return e2b.target(name, n, rng), dict()


# --------------------------------------------------------------------------- Leg A
def leg_a(name, n=64, budgets=(1, 2, 3, 4, 6, 8, 12, 16), seed=0, log=print):
    X, Y = _grid(n)
    rng = np.random.default_rng(seed)
    y, meta = make_target(name, n, rng)
    half = 0.5 * float(y @ y)
    lo, hi = theta_box(n)
    bank = dense_bank(n)
    pc = lambda e: 100.0 * e / half

    nmin, nmed, nmax, sub, n_narrow, n_wide = norm_spread(n, X, Y, lo, hi, rng)
    log(f"# target={name} n={n} 0.5||y||^2={half:.4f} |bank|={len(bank)} "
        f"Theta = positions x [sigma 1px..{n/4:.0f}px], |v|<={n/4:.0f}")
    log(f"# discrete atom norm over Theta: min={nmin:.1f} med={nmed:.1f} "
        f"max={nmax:.1f}, spread={nmax/nmin:.2f}x (analytic convention wants 1.00x)")
    log(f"#   narrow quartile median={n_narrow:.1f}, wide quartile median="
        f"{n_wide:.1f}; {100*sub:.1f}% of Theta has sub-pixel minor axis")

    Nmax = max(budgets)
    t0 = time.time()
    steps = greedy(y, n, X, Y, bank, Nmax, lo, hi, rng, budgets=budgets)
    log(f"# greedy to N={Nmax} in {time.time()-t0:.0f}s")

    # ---- N=1: the one budget where the (P0) optimum is computable outright
    s_y, _ = sup_corr(y, n, X, Y, bank, lo, hi, rng)
    s_nrm, _ = sup_corr(y, n, X, Y, bank, lo, hi, rng, normalize=True)
    E1_opt = half - 0.5 * s_nrm ** 2
    E1_greedy = steps[0][1]
    # greedy's N=1 solution is feasible, so E1_greedy >= E1_opt is forced. A
    # negative cost is therefore not a finding but a self-report that the
    # normalised argmax search fell short -- the same diagnostic role section 9.4
    # gives the negative duality gap.
    c0 = "" if E1_greedy >= E1_opt - 1e-9 * half else "   <- FAIL: search shortfall"
    log(f"# N=1 exact optimum {pc(E1_opt):.4f}% vs greedy {pc(E1_greedy):.4f}% "
        f"-> matched-filter placement costs {pc(E1_greedy-E1_opt):+.4f}pp "
        f"(0 iff the grid dictionary is exactly unit-norm){c0}")

    # ---- certified dual directions (s(y) already computed above, reused)
    dirs = [(y, s_y, float(y @ y), float(y @ y))]
    for (_, _, _, q) in steps:
        s, _ = sup_corr(q, n, X, Y, bank, lo, hi, rng)
        dirs.append((q, s, float(q @ q), float(q @ y)))
    # The optimal p interpolates between y (best at small M) and the greedy
    # residual (best at large M); B(.;M) is concave in p, so mixes can beat both.
    yn = y / np.linalg.norm(y)
    for (k, _, _, r) in steps:
        if k not in (1, 2, 4, 8, 16):
            continue
        rn = r / max(np.linalg.norm(r), 1e-300)
        for al in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            q = al * yn + (1 - al) * rn
            if q @ q <= 0:
                continue
            s, _ = sup_corr(q, n, X, Y, bank, lo, hi, rng,
                            n_refine=8, n_random=8000)
            dirs.append((q, s, float(q @ q), float(q @ y)))
    # Concave ascent at each reported budget's own mass. Iterates land in the
    # shared pool, so an ascent run for one M can also tighten another. The
    # before/after is reported because it separates the two ways the bound can
    # be loose: a poor search over p (fixable) versus the relaxation gap
    # itself (not fixable by this instrument).
    pre = {k: best_bound(dirs, y, Mg) for (k, _, Mg, _) in steps if k in budgets}
    t1 = time.time()
    for (k, _, Mg, _) in steps:
        if k in budgets:
            ascend(dirs, y, Mg, n, X, Y, bank, lo, hi, rng)
    gain = [100 * (best_bound(dirs, y, Mg) - pre[k]) / max(pre[k], 1e-300)
            for (k, _, Mg, _) in steps if k in budgets and pre[k] > 0]
    glo, ghi = (min(gain), max(gain)) if gain else (0.0, 0.0)
    # Read off the data, not asserted: a near-zero gain means the hand-picked
    # family was already at this instrument's ceiling, so what remains is the
    # relaxation gap. A large gain means the opposite, and says nothing about
    # whether further search would help again.
    verdict = ("hand-picked family was already at the ceiling, so the residual "
               "looseness is the relaxation gap" if ghi < 1.0 else
               "search over p was NOT saturated, so the residual looseness is "
               "not attributable to the relaxation gap alone")
    log(f"# {len(dirs)} dirs after ascent ({time.time()-t1:.0f}s); ascent improved "
        f"the bound by {glo:.1f}-{ghi:.1f}% -> {verdict}")

    log("")
    log("  greedy vs a certified lower bound on the (P0) optimum at mass <= M_g")
    log(f"{'N':>3} {'E_greedy':>9} {'M_g':>7} {'L(M_g)':>9} {'L/E':>6} "
        f"{'subopt<=':>9} {'s+5%':>8} {'s+10%':>8} {'s+25%':>8}  check")
    rows = []
    for (k, f, Mg, _) in steps:
        if k not in budgets:
            continue
        L = best_bound(dirs, y, Mg)
        c1 = "ok" if L <= f * (1 + 1e-9) else "FAIL-C1"
        # L=0 is only vacuous when greedy's error is not itself 0. On an
        # in-model target past full recovery, E_greedy=0 and L=0 is EXACT.
        tag = "" if L > 0 else ("  (exact: E_opt=0)" if f <= 1e-9 * half
                                else "  (vacuous)")
        log(f"{k:3d} {pc(f):9.4f} {Mg:7.3f} {pc(L):9.4f} {L/f:6.3f} "
            f"{100*(f-L)/f:8.1f}% {pc(best_bound(dirs,y,Mg,1.05)):8.4f} "
            f"{pc(best_bound(dirs,y,Mg,1.10)):8.4f} "
            f"{pc(best_bound(dirs,y,Mg,1.25)):8.4f}  {c1}{tag}")
        rows.append(dict(N=k, E=f, M=Mg, L=L, check=c1))

    log("")
    log("  mass sweep: the bound is conditional on M, so this is how fast it")
    log("  decays as the budget is relaxed beyond greedy's own mass")
    log(f"{'N':>3} {'E_greedy':>9} " + " ".join(
        f"{f'L({m}Mg)':>10}" for m in (1.0, 1.25, 1.5, 2.0, 3.0)))
    for (k, f, Mg, _) in steps:
        if k not in budgets:
            continue
        log(f"{k:3d} {pc(f):9.4f} " + " ".join(
            f"{pc(best_bound(dirs, y, mult*Mg)):10.4f}"
            for mult in (1.0, 1.25, 1.5, 2.0, 3.0)))

    if "M_true" in meta:
        Lt = best_bound(dirs, y, meta["M_true"])
        ok = "ok" if Lt <= 1e-6 * half else "FAIL-C2"
        log("")
        log(f"  C2 in-model control: ground truth attains error exactly 0 at mass "
            f"{meta['M_true']:.3f},")
        log(f"  so the bound there must be <= 0. Near-equal amplitudes make this "
            f"nearly tight.")
        log(f"  L(M_true) = {pc(Lt):.8f}% of 0.5||y||^2   -> {ok}")
        rows.append(dict(N=-1, check=ok))
    return rows


# --------------------------------------------------------------------------- Leg B
def leg_b(name, n=48, stride=2, seed=0, log=print):
    """Exact best-2 over an enumerable dictionary: greedy's myopia, measured."""
    X, Y = _grid(n)
    rng = np.random.default_rng(seed)
    y, _ = make_target(name, n, rng)
    half = 0.5 * float(y @ y)
    pc = lambda e: 100.0 * e / half

    sig_px = np.array([2.0, 3.5, 6.0])
    sig_px = sig_px[sig_px <= n / 4.0]
    shapes = shape_bank(np.log(sig_px / n), [0.45, 1.0],
                        np.linspace(0, np.pi, 4, endpoint=False))
    px = (np.arange(0, n, stride) + 0.5) / n
    PX, PY = np.meshgrid(px, px, indexing="ij")
    th = np.array([[cx, cy, u, v, w] for (u, v, w) in shapes
                   for (cx, cy) in zip(PX.ravel(), PY.ravel())])
    D = th.shape[0]

    G = np.empty((D, n * n), dtype=np.float32)
    for i in range(0, D, 2000):
        blk = atoms(th[i:i + 2000], X, Y)[0]
        G[i:i + 2000] = blk / np.linalg.norm(blk, axis=1, keepdims=True)
    yf = y.astype(np.float32)
    b = G @ yf
    log(f"# target={name} n={n} dictionary D={D} ({len(shapes)} shapes x "
        f"{PX.size} positions), {D*(D-1)//2:,} pairs enumerated exactly")

    def pair_stats(g, bi, bj):
        """Least-squares reduction and l1 mass for every pair, vectorised.

        Promoted to float64: den = 1-g^2 cancels catastrophically as |g|->1, and
        in float32 that noise would FABRICATE the very near-duplicate blow-up
        the 'free' column exists to expose."""
        g = np.asarray(g, dtype=np.float64)
        bi = np.asarray(bi, dtype=np.float64)
        bj = np.asarray(bj, dtype=np.float64)
        den = 1.0 - g * g
        bad = den < 1e-6
        dens = np.where(bad, 1.0, den)
        red = (bi * bi + bj * bj - 2.0 * g * bi * bj) / dens
        ai, aj = (bi - g * bj) / dens, (bj - g * bi) / dens
        mass = np.abs(ai) + np.abs(aj)
        red = np.where(bad, np.maximum(bi * bi, bj * bj), red)
        mass = np.where(bad, np.maximum(np.abs(bi), np.abs(bj)), mass)
        return red, mass, np.abs(g)

    best1 = float((b * b).max())
    i_star = int(np.argmax(np.abs(b)))
    g_star = G @ G[i_star]
    red_g, mass_g, _ = pair_stats(g_star, b[i_star], b)
    # EXCLUDE the self-pair; zeroing g_star[i_star] instead would fake an atom
    # orthogonal to itself and hand greedy a fictitious 2*best1 reduction.
    red_g[i_star] = -np.inf
    j_g = int(np.argmax(red_g))
    red_greedy, mass_greedy = float(red_g[j_g]), float(mass_g[j_g])

    # three exhaustive variants: unconstrained, coherence-capped, mass-matched
    best = {"free": (-np.inf, None), "coh<=0.9": (-np.inf, None),
            "mass<=greedy": (-np.inf, None)}
    for i in range(0, D, 256):
        Gi = (G[i:i + 256] @ G.T).astype(np.float32)
        red, mass, ag = pair_stats(Gi, b[i:i + 256, None], b[None, :])
        for r in range(Gi.shape[0]):
            red[r, i + r] = -np.inf
        for key, mask in (("free", None), ("coh<=0.9", ag <= 0.9),
                          ("mass<=greedy", mass <= mass_greedy)):
            rr = red if mask is None else np.where(mask, red, -np.inf)
            k = int(np.argmax(rr))
            v = float(rr.ravel()[k])
            if v > best[key][0]:
                best[key] = (v, (i + k // D, k % D, float(mass.ravel()[k]),
                                 float(ag.ravel()[k])))

    log(f"{'':>26}{'error':>10} {'% of 0.5||y||^2':>17} {'l1 mass':>9} {'coh':>6}")
    log(f"{'best-1 (greedy N=1)':>26}{half-0.5*best1:10.2f} "
        f"{pc(half-0.5*best1):16.4f}%")
    log(f"{'best-2 greedy':>26}{half-0.5*red_greedy:10.2f} "
        f"{pc(half-0.5*red_greedy):16.4f}% {mass_greedy:9.3f}")
    for key, (v, info) in best.items():
        log(f"{'best-2 exhaustive ' + key:>26}{half-0.5*v:10.2f} "
            f"{pc(half-0.5*v):16.4f}% {info[2]:9.3f} {info[3]:6.3f}")
    v_fair = best["mass<=greedy"][0]
    E_g, E_e = half - 0.5 * red_greedy, half - 0.5 * v_fair
    # C3: the exhaustive search contains greedy's own pair, so it can never come
    # out worse. A violation is an indexing or masking bug, not a finding.
    c3 = "ok" if red_greedy <= best["free"][0] * (1 + 1e-9) else "FAIL-C3"
    log(f"# C3 exhaustive-contains-greedy: greedy red={red_greedy:.2f} <= "
        f"free red={best['free'][0]:.2f} -> {c3}")
    log(f"# myopia cost at N=2, mass-matched: {100*(E_g-E_e)/E_e:+.3f}% of the "
        f"exhaustive optimum -> "
        f"{'greedy is exactly optimal here' if E_g <= E_e*(1+1e-9) else 'greedy is beaten'}")
    log(f"# greedy first atom idx {i_star}; unconstrained winner {best['free'][1][:2]} "
        f"at coherence {best['free'][1][3]:.4f}, mass {best['free'][1][2]:.2f} "
        f"(vs greedy mass {mass_greedy:.2f})")
    return dict(E_greedy2=E_g, E_exh2=E_e, D=D)


# ----------------------------------------------------------------------------- main
def main(targets=("inmodel", "cartoon"), n=64,
         budgets=(1, 2, 3, 4, 6, 8, 12, 16), out=None, do_b=True):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E3: certified lower bound on the (P0) optimum (Leg A), exact N=2")
    log("# myopia (Leg B). Module docstring states what each can and cannot show.")
    log(f"# params: n={n} budgets={budgets} targets={targets}")
    for t in targets:
        for name, fn in (("A", lambda: leg_a(t, n=n, budgets=budgets, log=log)),
                         ("B", lambda: leg_b(t, log=log))):
            if name == "B" and not do_b:
                continue
            log("")
            log("=" * 78)
            try:
                fn()
            except Exception:
                log(f"# leg {name} FAILED for {t}:")
                for ln in traceback.format_exc().splitlines():
                    log("#   " + ln)
    if out:
        with open(out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    tg = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("inmodel", "cartoon")
    main(targets=tg, out=sys.argv[2] if len(sys.argv) > 2 else None)
