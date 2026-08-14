"""E4: formulation versus solver, on a problem where BOTH are solved exactly.

Every comparison in sections 9 and 10 confounds two things. When the convex route
loses to greedy, that could mean the l1 FORMULATION is worse than l0, or merely
that the BLASSO SOLVER did not converge -- section 9.3 shows it demonstrably does
not past N=16, and section 10's own caveat says "a better-engineered BLASSO could
plausibly close or reverse the gap" (U7). Nothing in the document separates these.

This experiment removes the confound by shrinking the problem until every route
is solvable to optimality:

  * finite dictionary of D atoms on a small image, so no continuous search;
  * l0 solved by EXHAUSTIVE enumeration of all C(D,N) supports -- the true
    global optimum, not a heuristic;
  * l1 solved by the exact LARS/LASSO path, which is piecewise linear and
    therefore exact at every breakpoint, then debiased on its own support;
  * greedy run on the identical dictionary.

Whatever separates them here is the formulation, because no solver failed.

WHY THIS IS THE RIGHT SMALL PROBLEM. Two theoretical facts motivate it.

1. In measure space, convexification cannot see N at all. The extreme points of
   the TV ball {|m| <= tau} are the signed Diracs +-tau*delta_theta, which are
   1-sparse. By Krein-Milman the closed convex hull of the 1-sparse measures of
   mass <= tau is therefore the whole ball, so for EVERY N >= 1

       closed-conv{m : m has <= N atoms, |m|(Theta) <= tau} = {m : |m|(Theta) <= tau}.

   The atom count is annihilated by convexification. No convex penalty on
   measures -- not l1, not a reweighted or spatially varying variant -- can
   distinguish "N atoms" from "any number of atoms of the same total mass". The
   relaxation gap of section 2.1 is therefore structural, not a defect of the
   particular penalty chosen.

2. On a FINITE dictionary with bounded amplitudes the picture changes, because
   conv{c : ||c||_0 <= N, ||c||_inf <= M} = {c : ||c||_1 <= N*M, ||c||_inf <= M},
   which does depend on N. This is exactly the structure that exact l0
   branch-and-bound exploits, and it is why solvers like L0BnB reach p ~ 1e7 with
   ~20 nonzeros. The grid is not merely a computational convenience: by imposing
   a minimum separation it is what makes the sparsity constraint convex-
   representable at all. Section 8 says the recovery theory needs separation;
   this says the OPTIMIZATION needs it too, for a different reason.

WHAT THIS CAN SHOW. Whether l1 loses to l0 at matched N when neither solver is
at fault, by how much, and whether it picks a different support.

WHAT THIS CANNOT SHOW. N <= 4 on a few hundred atoms, one instance per target.
Exhaustive enumeration is C(D,N), so this cannot be pushed to section 10's N=8
or 16 -- reaching those needs branch-and-bound, which is the point of the
theory above, not something demonstrated here. Nothing here is off-the-grid, so
it says nothing about the continuous problem except through the dictionary.

A THIRD MEASUREMENT falls out. Section 10.2's lower bound is calibrated only at
N=1, the one budget where the optimum was known. Here the optimum is known for
every N reported, and on a finite dictionary the supremum s(p) = max_j |<g_j,p>|
is exact rather than searched -- so the bound becomes a theorem and its
looseness can be measured at N>1 for the first time.
"""

import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path

# LARS warns when the active set degenerates on a coherent dictionary. That is
# expected here and is itself part of the finding (see the coherence line in the
# output); it does not affect the breakpoints actually used.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from e2_relaxation_gap import _grid, atoms, shape_bank
import e2b_natural as e2b


# ------------------------------------------------------------------ dictionary
def build_dict(n, stride, sig_px, aspects, n_rot):
    X, Y = _grid(n)
    shapes = shape_bank(np.log(np.asarray(sig_px) / n), aspects,
                        np.linspace(0, np.pi, n_rot, endpoint=False))
    px = (np.arange(0, n, stride) + 0.5) / n
    PX, PY = np.meshgrid(px, px, indexing="ij")
    th = np.array([[cx, cy, u, v, w] for (u, v, w) in shapes
                   for (cx, cy) in zip(PX.ravel(), PY.ravel())])
    G = atoms(th, X, Y)[0]
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    # Deduplicate. Rotating an ISOTROPIC Gaussian reproduces it exactly, so a
    # shape bank crossed with rotations contains literal duplicates. Left in,
    # they waste enumeration slots, force the coherence to exactly 1, and
    # degenerate the LARS active set.
    keep, Gr = [], G @ G.T
    for j in range(G.shape[0]):
        if not keep or np.abs(Gr[j, keep]).max() < 1 - 1e-6:
            keep.append(j)
    return th[keep], G[keep], len(shapes), PX.size


def make_target(name, n, G, rng, K=3):
    if name == "inmodel":
        # true atoms drawn FROM the dictionary, so the l0 optimum at N=K is
        # exactly 0 and every method has a known answer to be judged against
        idx = rng.choice(G.shape[0], K, replace=False)
        c = rng.uniform(0.7, 1.3, K)
        return c @ G[idx], dict(K=K, support=set(idx.tolist()))
    return e2b.target(name, n, rng), dict()


# --------------------------------------------------------------- exact l0
def _extend(C, D):
    """All strictly-increasing (k+1)-tuples extending the k-tuples in C."""
    last = C[:, -1]
    cnt = D - 1 - last
    keep = cnt > 0
    C, last, cnt = C[keep], last[keep], cnt[keep]
    rep = np.repeat(np.arange(len(C)), cnt)
    start = np.cumsum(cnt) - cnt
    offs = np.arange(cnt.sum()) - np.repeat(start, cnt)
    return np.column_stack([C[rep], (last[rep] + 1 + offs)]).astype(np.int32)


def exact_l0(G, y, N, Gram, b, half, mass_cap=None, chunk=2_000_000, log=print):
    """Global optimum of the N-atom problem by exhaustive enumeration.

    Returns the best support and error, both unconstrained and (if mass_cap is
    given) restricted to solutions whose l1 amplitude mass is within the cap.
    The cap matters: with a coherent dictionary an unconstrained winner can be a
    near-duplicate pair with enormous opposing amplitudes -- correct arithmetic,
    useless as an encoding, and precisely what a box constraint forbids."""
    D = G.shape[0]
    from math import comb
    need = comb(D, N) * N * 4 / 2**30
    if need > 3.0:
        raise MemoryError(
            f"C({D},{N}) index array would need {need:.1f} GiB. Exhaustive "
            f"enumeration is the guarantee here, so this is a hard ceiling on "
            f"(D,N), not something to work around by sampling.")
    C = np.arange(D, dtype=np.int32)[:, None]
    for _ in range(N - 1):
        C = _extend(C, D)
    total = len(C)
    best = (-np.inf, None, None)
    best_cap = (-np.inf, None, None)
    for i in range(0, total, chunk):
        S = C[i:i + chunk]
        bS = b[S]                                           # (m,N)
        GS = Gram[S[:, :, None], S[:, None, :]]             # (m,N,N)
        GS = GS + 1e-10 * np.eye(N)                         # numerical only
        sol = np.linalg.solve(GS, bS[..., None])[..., 0]    # (m,N)
        expl = (bS * sol).sum(axis=1)
        mass = np.abs(sol).sum(axis=1)
        k = int(np.argmax(expl))
        if expl[k] > best[0]:
            best = (float(expl[k]), S[k].copy(), float(mass[k]))
        if mass_cap is not None:
            e2 = np.where(mass <= mass_cap, expl, -np.inf)
            k2 = int(np.argmax(e2))
            if e2[k2] > best_cap[0]:
                best_cap = (float(e2[k2]), S[k2].copy(), float(mass[k2]))
    out = dict(n_supports=total,
               err=half - 0.5 * best[0], support=best[1], mass=best[2])
    if mass_cap is not None:
        out.update(err_cap=half - 0.5 * best_cap[0], support_cap=best_cap[1],
                   mass_cap_used=best_cap[2])
    return out


# --------------------------------------------------------------- exact l1
def exact_lasso(G, y, N, half):
    """Exact LASSO at the breakpoint with N active atoms, then debiased.

    lars_path returns the piecewise-linear regularisation path, so the solution
    at each breakpoint is exact -- there is no optimisation error to blame."""
    _, _, coefs = lars_path(G.T, y, method="lasso")
    nnz = (np.abs(coefs) > 0).sum(axis=0)
    hit = np.where(nnz == N)[0]
    if len(hit) == 0:
        j = int(np.argmin(np.abs(nnz - N)))
    else:
        j = int(hit[-1])                       # least-regularised such point
    c = coefs[:, j]
    S = np.flatnonzero(c)
    r = y - c @ G
    raw = 0.5 * r @ r
    cd = np.linalg.lstsq(G[S].T, y, rcond=None)[0]          # debias on support
    rd = y - cd @ G[S]
    return dict(support=S, n_active=len(S), err_raw=raw,
                err=0.5 * rd @ rd, mass=float(np.abs(cd).sum()))


def greedy_grid(G, y, N, Gram, b, half):
    """Matching pursuit with full re-projection (OMP) on the same dictionary."""
    S = []
    for _ in range(N):
        r = y - (np.linalg.lstsq(G[S].T, y, rcond=None)[0] @ G[S] if S else 0.0)
        corr = np.abs(G @ r)
        corr[S] = -np.inf
        S.append(int(np.argmax(corr)))
    c = np.linalg.lstsq(G[S].T, y, rcond=None)[0]
    r = y - c @ G[S]
    return dict(support=np.array(S), err=0.5 * r @ r, mass=float(np.abs(c).sum()))


def dual_bound(G, y, M, n_dir=400):
    """Section 10.2's bound, here EXACT: on a finite dictionary the supremum
    s(p) = max_j |<g_j,p>| is a maximum over D numbers, not a search. So this is
    a theorem rather than an estimate, and its looseness against a known optimum
    is measurable at every N."""
    best = 0.0
    dirs = [y]
    S = []
    for _ in range(min(n_dir, 12)):            # greedy residuals as directions
        r = y - (np.linalg.lstsq(G[S].T, y, rcond=None)[0] @ G[S] if S else 0.0)
        dirs.append(r.copy())
        corr = np.abs(G @ r)
        if S:
            corr[S] = -np.inf
        S.append(int(np.argmax(corr)))
    for q in dirs:
        qq = float(q @ q)
        if qq <= 0:
            continue
        s = float(np.abs(G @ q).max())
        num = float(q @ y) - M * s
        if num > 0:
            best = max(best, num * num / (2 * qq))
    return best


# ------------------------------------------------------------------ experiment
def run(name, n=32, stride=4, sig_px=(2.0, 3.5, 6.0), aspects=(0.5, 1.0),
        n_rot=3, budgets=(1, 2, 3), seed=0, log=print):
    rng = np.random.default_rng(seed)
    th, G, n_sh, n_pos = build_dict(n, stride, sig_px, aspects, n_rot)
    D = G.shape[0]
    y, meta = make_target(name, n, G, rng, K=min(3, max(budgets)))
    half = 0.5 * float(y @ y)
    Gram, b = G @ G.T, G @ y

    log(f"# target={name} n={n} D={D} ({n_sh} shapes x {n_pos} positions) "
        f"0.5||y||^2={half:.4f}")
    coh = np.abs(Gram - np.eye(D)).max()
    log(f"# dictionary coherence max|<g_i,g_j>| = {coh:.4f}  "
        f"(l1 recovery theory wants this well below 1)")
    if meta:
        log(f"# in-model: {meta['K']} dictionary atoms, so the l0 optimum at "
            f"N={meta['K']} is exactly 0")

    pc = lambda e: 100.0 * e / half
    log("")
    log(f"{'N':>2} {'C(D,N)':>12} {'l0 exact':>9} {'greedy':>9} {'l1 debias':>10} "
        f"{'l1 raw':>9} | {'greedy-l0':>9} {'l1-l0':>8} | {'supp∩':>6} {'bound':>8}")
    rows = []
    for N in budgets:
        t0 = time.time()
        # both baselines first, so the mass cap is set wide enough that each is
        # feasible under it -- otherwise the capped optimum could "lose" to a
        # method the cap excluded, and check C4 would fire spuriously
        g = greedy_grid(G, y, N, Gram, b, half)
        l1 = exact_lasso(G, y, N, half)
        cap = 2.0 * max(g["mass"], l1["mass"])
        l0 = exact_l0(G, y, N, Gram, b, half, mass_cap=cap)
        e0, eg, e1 = l0["err_cap"], g["err"], l1["err"]
        ov = len(set(l0["support_cap"].tolist()) & set(l1["support"].tolist()))
        L = dual_bound(G, y, l0["mass_cap_used"])
        # excess over the exact optimum. When the optimum is 0 (in-model at
        # N=K) a ratio is meaningless, so report the absolute excess instead.
        tiny = 1e-9 * half
        rel = (lambda e: f"{100*(e-e0)/e0:8.2f}%" if e0 > tiny
               else f"{pc(e-e0):8.4f}pp")
        log(f"{N:2d} {l0['n_supports']:12,d} {pc(e0):9.4f} {pc(eg):9.4f} "
            f"{pc(e1):10.4f} {pc(l1['err_raw']):9.4f} | "
            f"{rel(eg):>9} {rel(e1):>8} | "
            f"{ov:2d}/{N:<3d} {pc(L):8.4f}   [{time.time()-t0:.0f}s]")
        rows.append(dict(N=N, l0=e0, greedy=eg, l1=e1, bound=L, overlap=ov,
                         n_active=l1["n_active"]))
        # C4: greedy's support and the LASSO support are both inside the
        # enumeration and both obey the mass cap, so neither can beat the
        # exhaustive optimum. A violation is an enumeration bug, not a finding.
        assert e0 <= eg + tiny and e0 <= e1 + tiny, \
            f"FAIL-C4: exhaustive l0 ({e0:.6g}) beaten by greedy ({eg:.6g}) " \
            f"or lasso ({e1:.6g})"
    log("")
    log("  greedy-l0 and l1-l0 are excess error over the exact global optimum, in %.")
    log("  supp∩ counts atoms the l1 support shares with the l0-optimal support.")
    log("  bound is section 10.2's certified lower bound, exact here, at the")
    log("  l0 optimum's own mass -- so l0 exact minus bound is pure slack.")
    return rows


def decorrelate(G, mu_max, seed=0):
    """Largest subdictionary found by a greedy sweep with pairwise coherence
    <= mu_max. The sweep order is a FIXED permutation, independent of the
    target, so the subdictionary is not selected using the answer."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(G.shape[0])
    keep = []
    for j in order:
        if not keep or np.abs(G[j] @ G[keep].T).max() <= mu_max:
            keep.append(int(j))
    return np.array(sorted(keep))


def coherence_sweep(name, n=32, N=3, mus=(0.99, 0.9, 0.75, 0.6, 0.45),
                    seed=0, log=print):
    """Does dictionary COHERENCE drive the l1-vs-l0 and greedy-vs-l0 gaps?

    Motivation, from section 4. Erb-Hangelbroek-Ron reach curvelet-optimal
    rates using a construction built on the CURVELET TILING -- a tight frame,
    hence of bounded redundancy and controlled coherence. So a structured,
    low-coherence SUBSET of the Gaussian dictionary already suffices
    asymptotically. Meanwhile every difficulty measured in this repo is a
    coherence symptom: l1 recovery conditions fail, branch-and-bound
    relaxations weaken, and greedy's first choice misleads.

    That suggests the lever is the dictionary, not the solver or the penalty.
    Here coherence is swept directly, holding target, budget and every method
    fixed, so it is the only variable.

    Reading it: the l0 column is expected to WORSEN as coherence falls, because
    a thinner dictionary genuinely approximates less well. The hypothesis is
    about the RELATIVE columns -- whether greedy and l1 close on the optimum."""
    rng = np.random.default_rng(seed)
    _, Gfull, _, _ = build_dict(n, 4, (2.0, 3.5, 6.0), (0.5, 1.0), 3)
    y, _ = make_target(name, n, Gfull, rng, K=3)
    half = 0.5 * float(y @ y)
    pc = lambda e: 100.0 * e / half
    log(f"# coherence sweep: target={name} n={n} N={N}, full D={Gfull.shape[0]}")
    log(f"{'mu_max':>7} {'D':>5} {'coh':>6} {'l0 exact':>9} {'greedy':>9} "
        f"{'l1 debias':>10} | {'greedy-l0':>9} {'l1-l0':>8} | {'supp∩':>6}")
    rows = []
    for mu in mus:
        idx = decorrelate(Gfull, mu)
        G = Gfull[idx]
        D = len(idx)
        if D < N + 1:
            continue
        Gram, b = G @ G.T, G @ y
        coh = float(np.abs(Gram - np.eye(D)).max())
        g = greedy_grid(G, y, N, Gram, b, half)
        l1 = exact_lasso(G, y, N, half)
        l0 = exact_l0(G, y, N, Gram, b, half,
                      mass_cap=2.0 * max(g["mass"], l1["mass"]))
        e0, eg, e1 = l0["err_cap"], g["err"], l1["err"]
        ov = len(set(l0["support_cap"].tolist()) & set(l1["support"].tolist()))
        log(f"{mu:7.2f} {D:5d} {coh:6.3f} {pc(e0):9.4f} {pc(eg):9.4f} "
            f"{pc(e1):10.4f} | {100*(eg-e0)/max(e0,1e-12):8.2f}% "
            f"{100*(e1-e0)/max(e0,1e-12):7.2f}% | {ov:2d}/{N:<3d}")
        rows.append(dict(mu=mu, D=D, coh=coh, l0=e0, greedy=eg, l1=e1))
    return rows


def build_parabolic(n, specs, alpha=1.0):
    """A curvelet-like Gaussian dictionary: parabolic scaling, orientations
    growing with anisotropy, and centres on a lattice ROTATED to match each
    atom, spaced proportionally to its own axes.

    This is the dictionary section 4's theorem actually uses. Erb-Hangelbroek-
    Ron build their N-term approximant on the curvelet tiling, whose defining
    features are exactly these: width ~ length^2, and a near-tiling rather than
    a dense overcomplete sweep. The coherence sweep showed that lowering
    coherence by RANDOM pruning closes the optimality gap but destroys
    approximation power. The question this answers is whether a DESIGNED
    low-coherence dictionary keeps both.

    specs: list of (length_px, width_px, n_orientations)."""
    X, Y = _grid(n)
    th = []
    for (L, W, n_rot) in specs:
        sL, sW = L / n, W / n                      # unit-square axes
        for t in np.linspace(0, np.pi, n_rot, endpoint=False):
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            A = np.linalg.inv(R @ np.diag([sL ** 2, sW ** 2]) @ R.T)
            M = np.linalg.cholesky(A).T
            # centres on a lattice aligned with this atom's own axes
            du, dv = alpha * sL, alpha * sW
            gu = np.arange(-1.0, 1.0 + du, du)
            gv = np.arange(-1.0, 1.0 + dv, dv)
            U, V = np.meshgrid(gu, gv, indexing="ij")
            P = np.stack([U.ravel(), V.ravel()], 1) @ R.T + 0.5
            P = P[(P[:, 0] >= 0) & (P[:, 0] <= 1) & (P[:, 1] >= 0) & (P[:, 1] <= 1)]
            for (cx, cy) in P:
                th.append([cx, cy, np.log(M[0, 0]), M[0, 1], np.log(M[1, 1])])
    th = np.array(th)
    G = atoms(th, X, Y)[0]
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    keep, Gr = [], G @ G.T
    for j in range(G.shape[0]):
        if not keep or np.abs(Gr[j, keep]).max() < 1 - 1e-6:
            keep.append(j)
    return th[keep], G[keep]


def build_dog(n, specs, alpha=2.5, k=1.6):
    """Same lattice, but each atom is a DIFFERENCE of two concentric Gaussians.

    Why this is the right control. The coherence sweep says coherence drives the
    optimality gap; the parabolic dictionary then showed that no lattice spacing
    lowers coherence, because the binding pairs are CROSS-SCALE -- a narrow
    Gaussian sitting inside a wide one has correlation ~0.9 by construction, and
    that is a property of positive bumps, not of their placement. Wavelets and
    curvelets escape it by oscillating (vanishing moments); a pure Gaussian
    cannot.

    A difference of two concentric Gaussians has zero mean, so it is orthogonal
    to constants and nearly orthogonal across scales. Crucially it stays INSIDE
    the 2D-splatting model: g_sigma - g_(k*sigma) is exactly two splats with
    opposite-signed coefficients, which GaussianImage's Eq. 7 already permits.
    So this changes the dictionary's geometry without leaving the model class,
    and tests whether the obstruction is the span or the parameterisation."""
    X, Y = _grid(n)
    # same lattice as build_parabolic; each centre becomes a concentric pair
    th_all, _ = build_parabolic(n, specs, alpha)
    G = []
    for t in th_all:
        inner = atoms(t[None, :], X, Y)[0][0]
        t2 = t.copy()
        t2[2] -= np.log(k)          # widen both axes by k  (e^u -> e^u / k)
        t2[4] -= np.log(k)
        t2[3] = t[3] / k
        outer = atoms(t2[None, :], X, Y)[0][0]
        d = inner / np.linalg.norm(inner) - outer / np.linalg.norm(outer)
        nz = np.linalg.norm(d)
        if nz > 1e-8:
            G.append(d / nz)
    G = np.array(G)
    keep, Gr = [], G @ G.T
    for j in range(G.shape[0]):
        if not keep or np.abs(Gr[j, keep]).max() < 1 - 1e-6:
            keep.append(j)
    return G[keep]


def evaluate(G, y, N, half, label, log):
    D = G.shape[0]
    Gram, b = G @ G.T, G @ y
    coh = float(np.abs(Gram - np.eye(D)).max())
    g = greedy_grid(G, y, N, Gram, b, half)
    l1 = exact_lasso(G, y, N, half)
    l0 = exact_l0(G, y, N, Gram, b, half,
                  mass_cap=2.0 * max(g["mass"], l1["mass"]))
    e0, eg, e1 = l0["err_cap"], g["err"], l1["err"]
    pc = lambda e: 100.0 * e / half
    log(f"{label:>22} {D:5d} {coh:6.3f} {pc(e0):9.4f} {pc(eg):9.4f} "
        f"{pc(e1):10.4f} | {100*(eg-e0)/max(e0,1e-12):8.2f}% "
        f"{100*(e1-e0)/max(e0,1e-12):7.2f}%")
    return dict(label=label, D=D, coh=coh, l0=e0, greedy=eg, l1=e1)


def designed_vs_random(name, n=32, N=3, seed=0, log=print):
    """Designed (parabolic) vs unstructured vs randomly-decorrelated, at
    matched N. The comparison that matters is ABSOLUTE achieved error, not the
    optimality gap: a thin dictionary trivially has a small gap because there is
    nothing left to get wrong."""
    rng = np.random.default_rng(seed)
    _, Gfull, _, _ = build_dict(n, 4, (2.0, 3.5, 6.0), (0.5, 1.0), 3)
    y, _ = make_target(name, n, Gfull, rng, K=3)
    half = 0.5 * float(y @ y)
    specs = [(12.0, 12.0, 1), (9.0, 6.0, 3), (6.5, 3.0, 6), (4.5, 1.6, 8)]
    _, Gpar = build_parabolic(n, specs, alpha=2.0)
    Gdog = build_dog(n, specs, alpha=2.0)
    log(f"# designed vs random: target={name} n={n} N={N}")
    log(f"{'dictionary':>22} {'D':>5} {'coh':>6} {'l0 exact':>9} {'greedy':>9} "
        f"{'l1 debias':>10} | {'greedy-l0':>9} {'l1-l0':>8}")
    out = [evaluate(Gfull, y, N, half, "gauss unstructured", log)]
    for mu in (0.90,):
        idx = decorrelate(Gfull, mu)
        out.append(evaluate(Gfull[idx], y, N, half, f"gauss decorr mu={mu}", log))
    out.append(evaluate(Gpar, y, N, half, "gauss parabolic", log))
    out.append(evaluate(Gdog, y, N, half, "DoG parabolic", log))
    bg = min(out, key=lambda r: r["greedy"])
    b1 = min(out, key=lambda r: r["l1"])
    log(f"# best ACHIEVED by greedy: {bg['label']} "
        f"({100*bg['greedy']/half:.4f}%);  by l1: {b1['label']} "
        f"({100*b1['l1']/half:.4f}%)")
    return out


def main(targets=("inmodel", "cartoon", "ascent", "face"), out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E4: exact l0 vs exact l1 at matched N, no solver error on either side.")
    log("# Panel A: richer dictionary, N<=3. Panel B: coarser one, N<=5.")
    log("# Exhaustive enumeration is C(D,N), so D and N trade off directly.")
    for t in targets:
        for tag, kw in (("A", dict(stride=4, sig_px=(2.0, 3.5, 6.0),
                                   aspects=(0.5, 1.0), n_rot=3,
                                   budgets=(1, 2, 3))),
                        ("B", dict(stride=8, sig_px=(2.5, 5.0),
                                   aspects=(0.5, 1.0), n_rot=3,
                                   budgets=(1, 2, 3, 4)))):
            log("")
            log("=" * 84)
            log(f"# panel {tag}")
            try:
                run(t, log=log, **kw)
            except Exception:
                import traceback
                log(f"# {t} panel {tag} FAILED:")
                for ln in traceback.format_exc().splitlines():
                    log("#   " + ln)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    tg = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else \
        ("inmodel", "cartoon", "ascent", "face")
    main(targets=tg, out=sys.argv[2] if len(sys.argv) > 2 else None)
