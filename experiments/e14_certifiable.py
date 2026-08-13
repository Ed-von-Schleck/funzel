"""E14 (U25): is ANY cheaply computable property of a solution a certificate?

Sections 10.8 and 10.9 close the relaxation route to a certificate: both standard
node bounds are loose on this dictionary, for a reason specific to it. Section
10.12 closes the empirical substitute: agreement between independent restarts
does not indicate optimality, and by N=8 there is nothing left to agree on. U25
is what remains -- whether some property of a SOLUTION ITSELF says how far that
solution is from the global optimum.

WHY THE TEST HAS TO BE CROSS-IMAGE, and this is the whole design. Within a single
image the error already ranks solutions perfectly, so a feature that merely
correlates with error inside one image is strictly worse than the number we
already have. What a certificate provides is an ABSOLUTE statement: given one
solution and no knowledge of the optimum, is this it? So the headline metric is
the pooled cross-image AUC computed on RAW, un-normalised feature values -- can
the feature separate optimal from suboptimal solutions when the images, and
therefore the difficulty levels, are mixed together? The error itself cannot do
this (8% may be optimal on one image and poor on another), and its failure is the
baseline the features have to beat.

Within-image AUC is reported too, but only as a diagnostic: it says whether a
feature carries any signal at all. It cannot make a feature useful, because the
error's within-image AUC is 1.0 by construction.

MEASURED AGAINST THE TRUE OPTIMUM, which is why this is at N=3. The optimum comes
from exhaustive enumeration of all C(D,3) supports, so "is this solution the
global optimum" is a fact rather than an estimate.

THREE POPULATIONS OF NEGATIVES, because the difficulty of the discrimination is
the whole story and one population hides it:

  rand   uniformly drawn supports. The easy case. A feature that cannot beat
         these carries no information about optimality at all.
  local  every 1-swap local optimum reachable by descent from random supports.
         The realistic case -- what a grid search actually returns. It turns out
         to be a small set: 2-5 per image, saturating by ~100 restarts, and
         descent reaches the true optimum on every image tested. That is worth
         recording on its own, and it means this population is too small to
         carry the statistics, so it is reported but not leaned on.
  top    the 200 lowest-error supports other than the optimum. The adversarial
         case, and the one a certificate has to survive: these are the solutions
         a certificate would be asked to rule out.

PART D carries the realism the grid cannot. Section 10.12 showed the continuous
problem has 16-60 distinct optima at N=3 where the grid has 2-5, so off-grid is
where the question actually bites -- but off-grid there is no proven optimum, so
the best of many restarts stands in for it and the leg is confirmatory only.

TWELVE FEATURES, chosen before running, each dimensionless so that pooling across
images is meaningful:

  cos_next     largest cosine between the residual and any unused atom. The most
               theoretically motivated: it is the lambda=0 dual certificate value
               of section 6, and the hypothesis is that a suboptimal solution
               leaves more exploitable structure behind.
  swap_margin  how much worse the best available single swap is, relative to the
               error. Basin depth. Costs O(N*D) solves, still cheap.
  coh          largest coherence among the chosen atoms.
  log_cond     conditioning of their Gram.
  amp_ratio    smallest to largest atom energy -- a wasted atom reads as near 0.
  eff_n        effective fraction of atoms carrying energy.
  resid_peak   peak residual over its rms -- a missed feature reads as a spike.
  resid_rough  residual roughness, a whiteness proxy.
  scale_spread spread of log atom widths.
  min_sep      closest pair of centres over the mean atom width (section 2.2's
               separation, measured on the solution).
  neg_frac     fraction of negative amplitudes.
  rand         a deterministic pseudo-random number. NULL CONTROL -- if this
               scores away from 0.5 the harness is broken.

HELD OUT, because twelve features on forty images will produce a winner by
chance. The images are split in half; the best feature AND its orientation are
chosen on one half and scored on the other. A logistic regression over all twelve
is scored the same way, so the conclusion covers linear combinations and not just
single features.

WHAT THIS CANNOT SHOW. Twelve hand-chosen features at N=3 on one dictionary
family. A null result does not prove no computable certificate exists; it bounds
what this class of cheap solution-intrinsic quantities can do. A POSITIVE result
would need re-checking on a fresh image set before being believed, which is
pre-registered here as the response to any held-out AUC above 0.9.
"""

import sys
import time

import numpy as np
from scipy.optimize import minimize

from e2_relaxation_gap import _grid, atoms
import e4_exact_l0 as e4
import e11_canonical as e11
import e12_canonical_offgrid as e12

FAIL = []


def check(name, ok, detail=""):
    print(f"    [{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- image set
def _cartoon(n, rng):
    """e2b's cartoon with its constants randomised, so the set has variety."""
    X, Y = _grid(n)
    x, y = X.reshape(n, n), Y.reshape(n, n)
    a, b = rng.uniform(1.5, 4.0, 2)
    s1 = 0.35 + 0.30 * np.cos(a * x + rng.uniform(0, 3)) * \
        np.cos(b * y - rng.uniform(0, 3))
    s2 = 0.85 - rng.uniform(0.2, 0.6) * (x - rng.uniform(0.2, 0.8)) ** 2 \
        - rng.uniform(0.1, 0.4) * y
    t = np.arctan2(y - 0.5, x - 0.5)
    rad = rng.uniform(0.18, 0.32) + 0.06 * np.cos(3 * t) + 0.03 * np.sin(2 * t)
    cx, cy = rng.uniform(0.35, 0.65, 2)
    inside = ((x - cx) ** 2 + (y - cy) ** 2) < rad ** 2
    return np.where(inside, s2, s1).ravel()


def image_set(n, rng, n_nat=28, n_cart=12, min_std=0.08, log=print):
    """Forty 48x48 images of varying difficulty: crops of two photographs at two
    zoom levels, plus randomised cartoons. Variety is the point -- the pooled
    test is only meaningful if the images differ in how hard they are."""
    import scipy.datasets as sd
    src = [sd.ascent().astype(float), sd.face(gray=True).astype(float)]
    out, rejected = [], 0
    while len(out) < n_nat:
        img = src[len(out) % 2]
        k = 1 if len(out) % 4 < 2 else 2          # native crop, or 2x downsampled
        m = n * k
        i = rng.integers(0, img.shape[0] - m)
        j = rng.integers(0, img.shape[1] - m)
        p = img[i:i + m, j:j + m]
        if k > 1:
            p = p.reshape(n, k, n, k).mean(axis=(1, 3))
        p = p - p.min()
        if p.max() <= 0:
            rejected += 1
            continue
        p = p / p.max()
        if p.std() < min_std:
            rejected += 1
            continue
        out.append((f"{'ascent' if len(out) % 2 == 0 else 'face'}x{k}", p.ravel()))
    for _ in range(n_cart):
        y = _cartoon(n, rng)
        y = y - y.min()
        out.append(("cartoon", (y / y.max()).ravel()))
    log(f"  {len(out)} images ({n_nat} photo crops, {n_cart} cartoons), "
        f"{rejected} rejected as too flat (std < {min_std})")
    return out


# ------------------------------------------------------- supports and errors
def err_batch(S, Gram, b, half, N):
    """Exact error of each support in S. Same solve as e11.all_errors."""
    bS = b[S]
    GS = Gram[S[:, :, None], S[:, None, :]] + 1e-10 * np.eye(N)
    sol = np.linalg.solve(GS, bS[..., None])[..., 0]
    return half - 0.5 * (bS * sol).sum(axis=1)


def neighbours(S, D):
    """Every single-atom swap of support S. Shape (N*(D-N), N)."""
    N = len(S)
    others = np.setdiff1d(np.arange(D, dtype=np.int32), S)
    out = np.repeat(S[None, :], N * len(others), axis=0)
    for i in range(N):
        out[i * len(others):(i + 1) * len(others), i] = others
    return out


def descend(S, Gram, b, half, D, N, max_steps=60):
    """Best-improvement 1-swap descent. Returns a 1-swap local optimum."""
    e = float(err_batch(S[None, :], Gram, b, half, N)[0])
    for _ in range(max_steps):
        nb = neighbours(S, D)
        en = err_batch(nb, Gram, b, half, N)
        k = int(np.argmin(en))
        if en[k] >= e - 1e-14 * max(abs(e), 1.0):
            return np.sort(S), e
        S, e = np.sort(nb[k]), float(en[k])
    return np.sort(S), e


def local_optima(Gram, b, half, D, N, rng, n_starts):
    """Distinct 1-swap local optima reached from random supports."""
    found = {}
    for _ in range(n_starts):
        S0 = np.sort(rng.choice(D, N, replace=False).astype(np.int32))
        S, e = descend(S0, Gram, b, half, D, N)
        found[tuple(S.tolist())] = e
    return found


# ------------------------------------------------------------------ features
def _shape_feats(thS, n):
    """Features of the atom geometry alone -- shared by grid and off-grid."""
    P = thS[:, :2] * n
    width = np.exp(-0.5 * (thS[:, 2] + thS[:, 4])) * n      # geometric-mean sigma
    Dm = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(Dm, np.inf)
    return dict(scale_spread=float(np.std(np.log(width))),
                min_sep=float(Dm.min() / max(width.mean(), 1e-12)))


def features(GS, thS, c, y, n, Gfull, rownorm, in_support, swap_margin, rnd):
    """All twelve, from the fitted solution. Every one dimensionless."""
    r = y - c @ GS
    rn = float(np.linalg.norm(r))
    # cos_next: the lambda=0 certificate value over the unused dictionary
    corr = np.abs(Gfull @ r) / rownorm
    corr[in_support] = -np.inf
    cos_next = float(corr.max() / max(rn, 1e-300))
    # coherence and conditioning of the chosen atoms
    gn = np.linalg.norm(GS, axis=1)
    Gn = GS / gn[:, None]
    M = np.abs(Gn @ Gn.T)
    np.fill_diagonal(M, 0.0)
    ev = np.linalg.eigvalsh(Gn @ Gn.T)
    # energy carried by each atom, so amplitude features are norm-independent
    a = np.abs(c) * gn
    R = r.reshape(n, n)
    rough = (np.diff(R, axis=0) ** 2).sum() + (np.diff(R, axis=1) ** 2).sum()
    f = dict(
        cos_next=cos_next,
        swap_margin=swap_margin,
        coh=float(M.max()),
        log_cond=float(np.log10(max(ev.max(), 1e-300) / max(ev.min(), 1e-12))),
        amp_ratio=float(a.min() / max(a.max(), 1e-300)),
        eff_n=float(a.sum() ** 2 / (len(a) * max((a * a).sum(), 1e-300))),
        resid_peak=float(np.abs(r).max() / max(rn / np.sqrt(len(r)), 1e-300)),
        resid_rough=float(rough / max(rn * rn, 1e-300)),
        neg_frac=float((c < 0).mean()),
        # drawn from a dedicated stream, so it is independent of the support's
        # index and of everything else -- a value derived from the enumeration
        # order would inherit whatever bias the optimum's atom indices carry
        rand=float(rnd),
    )
    f.update(_shape_feats(thS, n))
    return f


FEATS = ["cos_next", "swap_margin", "coh", "log_cond", "amp_ratio", "eff_n",
         "resid_peak", "resid_rough", "scale_spread", "min_sep", "neg_frac",
         "rand"]


# ------------------------------------------------------------------- scoring
def auc(pos, neg):
    """P(feature lower on an optimum than on a suboptimum), ties counted half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):                             # average ranks within ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = r[:len(pos)].sum()
    u = rp - len(pos) * (len(pos) + 1) / 2.0
    return 1.0 - u / (len(pos) * len(neg))         # low value on optima -> >0.5


def pooled_auc(rows, key, images=None):
    if images is not None:
        rows = [w for w in rows if w["img"] in images]
    return auc([w[key] for w in rows if w["opt"]],
               [w[key] for w in rows if not w["opt"]])


def within_auc(rows, key):
    vals = []
    for im in sorted({w["img"] for w in rows}):
        sub = [w for w in rows if w["img"] == im]
        a = auc([w[key] for w in sub if w["opt"]],
                [w[key] for w in sub if not w["opt"]])
        if not np.isnan(a):
            vals.append(a)
    return float(np.mean(vals))


def boot_ci(rows, key, rng, n_boot=400):
    imgs = sorted({w["img"] for w in rows})
    by = {im: [w for w in rows if w["img"] == im] for im in imgs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(imgs), len(imgs), replace=True)
        sub = [w for k in pick for w in by[imgs[k]]]
        a = auc([w[key] for w in sub if w["opt"]],
                [w[key] for w in sub if not w["opt"]])
        if not np.isnan(a):
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def precision_at_full_recall(rows, key, sign):
    """Threshold that flags EVERY optimum: what fraction of flags are optima?

    This is the certificate-shaped number -- a certificate must never miss the
    optimum, so recall is fixed at 1 and precision is what is left to measure."""
    v = np.array([sign * w[key] for w in rows])
    o = np.array([w["opt"] for w in rows])
    t = v[o].max()
    flagged = v <= t
    return float(o[flagged].sum() / flagged.sum()), int(flagged.sum()), len(v)


def logistic_holdout(rows, fit_imgs, test_imgs, feats):
    """L2 logistic regression, standardised and class-weighted on the fit half."""
    def mat(imgs):
        sub = [w for w in rows if w["img"] in imgs]
        return (np.array([[w[f] for f in feats] for w in sub]),
                np.array([w["opt"] for w in sub], float), sub)
    Xf, yf, _ = mat(fit_imgs)
    Xt, yt, subt = mat(test_imgs)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
    Xf, Xt = (Xf - mu) / sd, (Xt - mu) / sd
    wpos = len(yf) / max(2 * yf.sum(), 1.0)
    wneg = len(yf) / max(2 * (len(yf) - yf.sum()), 1.0)
    w = np.where(yf > 0, wpos, wneg)

    def obj(b):
        z = Xf @ b[1:] + b[0]
        ll = np.sum(w * (np.logaddexp(0, z) - yf * z))
        return ll + 1.0 * b[1:] @ b[1:]

    b = minimize(obj, np.zeros(Xf.shape[1] + 1), method="L-BFGS-B").x
    s = Xt @ b[1:] + b[0]
    return auc(-s[yt > 0], -s[yt == 0]), b


# ------------------------------------------------------------- A: grid legs
def run_grid(n=48, N=3, n_nat=28, n_cart=12, n_starts=100, k_top=200,
             k_rand=200, seed=0, log=print):
    rng = np.random.default_rng(seed)
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    D = len(G)
    Gram = G @ G.T
    rownorm = np.linalg.norm(G, axis=1)
    imgs = image_set(n, rng, n_nat=n_nat, n_cart=n_cart, log=log)
    log(f"  dictionary D={D}, N={N}, all C(D,{N})={D*(D-1)*(D-2)//6:,} supports "
        f"enumerated per image")

    rows, stats = [], []
    nullrng = np.random.default_rng(99)
    t0 = time.time()
    for ii, (nm, y) in enumerate(imgs):
        half = 0.5 * float(y @ y)
        b = G @ y
        C, errs = e11.all_errors(G, y, N, Gram, b, half)
        order = np.argsort(errs)
        istar = int(order[0])
        Sstar = np.sort(C[istar])
        estar = float(errs[istar])
        # the enumerated optimum must itself be 1-swap locally optimal
        if ii == 0:
            en = err_batch(neighbours(Sstar, D), Gram, b, half, N)
            check("C9 enumerated optimum is 1-swap locally optimal",
                  bool(en.min() >= estar - 1e-12),
                  f"(best neighbour {100*en.min()/half:.4f}% vs "
                  f"{100*estar/half:.4f}%)")
            # the off-grid margin is computed by a different route (leave one
            # out, refit against every candidate) and must agree with the
            # exhaustive swap on a support where both are defined
            m1 = float((en.min() - estar) / estar)
            m2 = replace_margin(G[Sstar], y, estar, half, G, b,
                                (G * G).sum(1), exclude=Sstar)
            check("C14 the two swap-margin implementations agree",
                  abs(m1 - m2) < 1e-8 * max(1.0, abs(m1)),
                  f"({m1:.8f} vs {m2:.8f})")
        # ---- the three negative populations, plus the one positive
        found = local_optima(Gram, b, half, D, N, rng, n_starts)
        reached = found.pop(tuple(Sstar.tolist()), None) is not None
        pops = {"opt": [Sstar]}
        pops["local"] = [np.array(s, dtype=np.int32) for s in sorted(found)]
        pops["top"] = [np.sort(C[i]) for i in order[1:k_top + 1]]
        pick = rng.choice(len(C), k_rand + 1, replace=False)
        pops["rand"] = [np.sort(C[i]) for i in pick if i != istar][:k_rand]
        for pop, sups in pops.items():
            for S in sups:
                GS = G[S]
                c = np.linalg.solve(Gram[S[:, None], S[None, :]]
                                    + 1e-10 * np.eye(N), b[S])
                e = float(err_batch(S[None, :], Gram, b, half, N)[0])
                en = err_batch(neighbours(S, D), Gram, b, half, N)
                in_sup = np.zeros(D, bool)
                in_sup[S] = True
                f = features(GS, th[S], c, y, n, G, rownorm, in_sup,
                             float((en.min() - e) / max(e, 1e-300)),
                             nullrng.random())
                f.update(img=ii, name=nm, kind=nm.split("x")[0], pop=pop,
                         err=e, half=half, err_pc=100.0 * e / half,
                         gap=float((e - estar) / estar),
                         opt=bool(pop == "opt"))
                rows.append(f)
        stats.append(dict(name=nm, n_local=len(found) + 1, reached=reached,
                          opt_pc=100 * estar / half,
                          top_gap=100 * (errs[order[k_top]] - estar) / estar))
    log(f"  {len(rows)} solutions over {len(imgs)} images "
        f"[{time.time()-t0:.0f}s]")
    npos = sum(1 for w in rows if w["opt"])
    check("C10 exactly one optimum per image", npos == len(imgs),
          f"({npos} positives, {len(imgs)} images)")
    # every LOCAL solution is 1-swap optimal, so no swap can improve it; a
    # negative margin would mean the descent returned a non-local-optimum
    loc = [w for w in rows if w["pop"] in ("local", "opt")]
    worst = min(w["swap_margin"] for w in loc)
    check("C11 no local optimum has an improving swap", worst >= -1e-12,
          f"(most negative margin {worst:.3e})")
    gmin = min(w["gap"] for w in rows)
    check("C12 no solution beats the enumerated optimum", gmin >= -1e-12,
          f"(most negative gap {gmin:.3e})")
    bad = [w for w in rows for k in FEATS if not np.isfinite(w[k])]
    check("C13 all features finite", not bad, f"({len(bad)} non-finite)")

    nloc = [s["n_local"] for s in stats]
    log(f"  1-swap local optima per image: median {np.median(nloc):.0f}, "
        f"range {min(nloc)}-{max(nloc)}; descent from {n_starts} random starts "
        f"reached the true optimum on "
        f"{sum(s['reached'] for s in stats)}/{len(stats)} images")
    log(f"  optimum error across images: "
        f"{min(s['opt_pc'] for s in stats):.2f}%.."
        f"{max(s['opt_pc'] for s in stats):.2f}% "
        f"-- the spread the pooled test has to see through")
    log(f"  the {k_top}th-best support is "
        f"{np.median([s['top_gap'] for s in stats]):.1f}% worse than the "
        f"optimum (median over images)")
    return rows, imgs


def replace_margin(GS, y, e0, half, G, Gy, Gsq, exclude=None):
    """Best error from replacing ONE fitted atom by a dictionary atom, relative
    to the current error. Positive means no such replacement improves.

    On the grid this is exactly the swap neighbourhood. Off-grid the fitted
    atoms are not dictionary elements, so this is the nearest well-defined
    analogue and is computed the same way: leave one out, refit with each
    candidate in its place."""
    N, D = len(GS), len(G)
    # a candidate identical to the atom being removed reproduces the current
    # solution exactly, so leaving those in would report a margin of zero for
    # every on-grid support and make the feature vacuous
    mask = np.zeros(D, bool)
    if exclude is not None:
        mask[exclude] = True
    best = np.inf
    for i in range(N):
        Gk = np.delete(GS, i, axis=0)
        A, B, bk = Gk @ Gk.T, Gk @ G.T, Gk @ y
        M = np.empty((D, N, N))
        M[:, :N - 1, :N - 1] = A
        M[:, :N - 1, N - 1] = B.T
        M[:, N - 1, :N - 1] = B.T
        M[:, N - 1, N - 1] = Gsq
        M += 1e-10 * np.eye(N)
        rhs = np.empty((D, N))
        rhs[:, :N - 1] = bk
        rhs[:, N - 1] = Gy
        sol = np.linalg.solve(M, rhs[..., None])[..., 0]
        cand = half - 0.5 * (rhs * sol).sum(1)
        cand[mask] = np.inf
        best = min(best, float(cand.min()))
    return (best - e0) / max(e0, 1e-300)


def vs(rows, pop):
    """The positives plus exactly one negative population."""
    return [w for w in rows if w["opt"] or w["pop"] == pop]


def report(rows, rng, pops=("rand", "local", "top"), main=None, log=print):
    """`main` is the population the held-out and threshold legs are run on."""
    imgs = sorted({w["img"] for w in rows})
    pops = list(pops)
    main = main or pops[-1]
    log("")
    log("  ## A. Can any feature separate the optimum from each population,")
    log("  ##    ACROSS images, on raw values? (>0.5 = lower on the optimum)")
    log(f"      {'feature':>13} " +
        " ".join(f"{'vs ' + p:>11}" for p in pops) +
        f" {main+': 95% CI':>18} {main+': within':>14}")
    for k in FEATS + ["err_pc", "gap"]:
        cells = " ".join(f"{pooled_auc(vs(rows, p), k):11.3f}" for p in pops)
        lo, hi = boot_ci(vs(rows, main), k, rng)
        w = within_auc(vs(rows, main), k)
        note = {"gap": "  <- CHEATING control, must be 1.00",
                "rand": "  <- NULL control, must be ~0.50",
                "err_pc": "  <- the error itself: the baseline"}.get(k, "")
        log(f"      {k:>13} {cells} {f'[{lo:.3f},{hi:.3f}]':>18} "
            f"{w:14.3f}{note}")
    check("C6 cheating feature scores 1.00",
          abs(pooled_auc(vs(rows, main), "gap") - 1.0) < 1e-9)
    lo, hi = boot_ci(vs(rows, main), "rand", rng)
    check("C7 null control's CI contains chance", lo <= 0.5 <= hi,
          f"(AUC {pooled_auc(vs(rows, main), 'rand'):.3f}, "
          f"CI [{lo:.3f},{hi:.3f}])")
    check("C8 the error ranks perfectly WITHIN an image",
          abs(within_auc(vs(rows, main), "err_pc") - 1.0) < 1e-9)

    log("")
    log(f"  ## A2. Is any of it a composition artefact? The '{main}' test rerun")
    log("  ##     within one image kind, so photo-vs-cartoon cannot contribute.")
    kinds = sorted({w["kind"] for w in rows})
    top_rows = vs(rows, main)
    log(f"      {'feature':>13} " + " ".join(f"{k:>12}" for k in kinds))
    for k in FEATS:
        cells = [f"{pooled_auc([w for w in top_rows if w['kind'] == kd], k):12.3f}"
                 for kd in kinds]
        log(f"      {k:>13} " + " ".join(cells))

    log("")
    log(f"  ## B. Held out on the '{main}' population: choose the")
    log("  ##    feature on half the images, score it on the other half.")
    # split on parity, not on position: the image list is photos then cartoons,
    # so a contiguous split would put a different MIX in each half and confound
    # "the feature does not generalise" with "the two halves are different sets"
    fit, test = set(imgs[0::2]), set(imgs[1::2])
    # the null control is not a candidate -- it is there to check the harness
    cands = [k for k in FEATS if k != "rand"]
    best = max(cands, key=lambda k: abs(pooled_auc(top_rows, k, fit) - 0.5))
    a_fit = pooled_auc(top_rows, best, fit)
    sign = 1.0 if a_fit >= 0.5 else -1.0
    a_test = pooled_auc(top_rows, best, test)
    a_test = a_test if sign > 0 else 1 - a_test
    log(f"      best on the fit half: {best} (AUC {max(a_fit, 1-a_fit):.3f}, "
        f"orientation {'low=optimal' if sign > 0 else 'high=optimal'})")
    log(f"      the same feature on the held-out half: AUC {a_test:.3f}")
    a_log, coef = logistic_holdout(top_rows, fit, test, FEATS)
    log(f"      logistic regression over all {len(FEATS)} features, "
        f"held out: AUC {a_log:.3f}")
    top = sorted(zip(FEATS, coef[1:]), key=lambda t: -abs(t[1]))[:3]
    log("      its largest weights: " +
        ", ".join(f"{k} {v:+.2f}" for k, v in top))

    log("")
    log("  ## C. The certificate-shaped question: a threshold that flags EVERY")
    log("  ##    optimum -- what fraction of what it flags is actually optimal?")
    log(f"      {'population':>10} {'feature':>13} {'precision':>10} "
        f"{'flagged':>14} {'base rate':>10}")
    for pop in pops:
        sub = vs(rows, pop)
        base = sum(w["opt"] for w in sub) / len(sub)
        # the raw error belongs in this table: off-grid it scores the highest
        # pooled AUC of anything measured, so "just threshold the error" is the
        # obvious objection and deserves exactly the same test
        for k in dict.fromkeys([best, "cos_next", "swap_margin", "err_pc"]):
            s = 1.0 if pooled_auc(sub, k) >= 0.5 else -1.0
            p, nf, nt = precision_at_full_recall(sub, k, s)
            log(f"      {pop:>10} {k:>13} {p:10.3f} {f'{nf}/{nt}':>14} "
                f"{base:10.3f}")
    return dict(best=best, sign=sign, a_test=a_test, a_log=a_log)


# ------------------------------------------------- D: off-grid confirmation
def run_continuous(n=48, N=3, n_nat=8, n_cart=4, n_restarts=60, seed=3,
                   log=print):
    rng = np.random.default_rng(seed)
    X, Y = _grid(n)
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    rownorm = np.linalg.norm(G, axis=1)
    Gsq = (G * G).sum(1)
    imgs = image_set(n, rng, n_nat=n_nat, n_cart=n_cart, log=log)
    log(f"  {n_restarts} continuous restarts on each of {len(imgs)} images. "
        f"Every solution here is a genuine local optimum of the real problem,")
    log("  which is what the grid could not supply. The reference is the best "
        "restart, NOT a proven optimum -- section 10.12 showed it is not one.")
    rows = []
    nullrng = np.random.default_rng(98)
    t0 = time.time()
    for ii, (nm, y) in enumerate(imgs):
        half = 0.5 * float(y @ y)
        Gy = G @ y
        sols = []
        for s in range(n_restarts):
            g = np.random.default_rng(4000 + 91 * ii + s)
            th0 = np.column_stack([
                g.uniform(0.05, 0.95, N), g.uniform(0.05, 0.95, N),
                np.log(g.uniform(2.0, 12.0, N)), np.zeros(N),
                np.log(g.uniform(2.0, 12.0, N))])
            A = atoms(th0, X, Y)[0]
            amp0 = np.linalg.lstsq(A.T, y, rcond=None)[0]
            e, thp = e12.polish(th0, amp0, y, X, Y, n)
            sols.append((e, thp))
        ebest = min(s[0] for s in sols)
        for (e, thp) in sols:
            GS = atoms(thp, X, Y)[0]
            c = np.linalg.lstsq(GS.T, y, rcond=None)[0]
            # nothing is excluded from the cos_next scan: the fitted atoms are
            # off-grid, and least squares already makes the residual orthogonal
            # to them, so a near-duplicate grid atom scores near zero anyway
            f = features(GS, thp, c, y, n, G, rownorm, np.zeros(len(G), bool),
                         replace_margin(GS, y, e, half, G, Gy, Gsq),
                         nullrng.random())
            f.update(img=ii, name=nm, kind=nm.split("x")[0], pop="restart",
                     err=e, half=half, err_pc=100.0 * e / half,
                     gap=float((e - ebest) / max(ebest, 1e-300)),
                     opt=bool(e <= ebest * (1 + 1e-6)))
            rows.append(f)
        nb = sum(1 for s in sols if s[0] <= ebest * (1 + 1e-6))
        log(f"      {nm:9s} best {100*ebest/half:7.4f}%, reached by "
            f"{nb:2d}/{n_restarts} restarts, median restart "
            f"{100*np.median([s[0] for s in sols])/half:7.4f}%")
    log(f"  [{time.time() - t0:.0f}s]")
    return rows


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E14 (U25): is any cheap property of a solution a certificate?")
    log("# The question is CROSS-IMAGE separation on raw feature values. A")
    log("# feature that only ranks within one image is beaten by the error.")
    log("")
    log("=" * 78)
    log("# GRID LEGS: the optimum is known exactly, by enumeration.")
    rows, _ = run_grid(log=log)
    report(rows, np.random.default_rng(11), log=log)
    log("")
    log("  READ THE 'vs top' COLUMN WITH CARE. Most of the 200 lowest-error")
    log("  supports are not even 1-swap locally optimal, so swap_margin scores")
    log("  near-perfectly there by detecting local optimality rather than")
    log("  global optimality -- and it collapses against the 'local' column,")
    log("  which is the population any real procedure actually returns. The")
    log("  grid has only 2-5 local optima per image, too few to measure on,")
    log("  which is what part D is for.")
    log("")
    log("=" * 78)
    log("# D. OFF-GRID: every solution is a real local optimum, and there are")
    log("#    enough of them. Reference is best-of-restarts, not proven optimal.")
    crows = run_continuous(log=log)
    res = report(crows, np.random.default_rng(12), pops=("restart",), log=log)

    # The pre-registered trigger for this leg was a held-out AUC above 0.9. It
    # runs unconditionally anyway: one split of forty images is a single noisy
    # measurement whichever way it lands, and a second independent image set
    # costs one run and says how much the held-out number itself moves.
    log("")
    log("=" * 78)
    log(f"# E. CONFIRMATION on a fresh image set. '{res['best']}' won the fit")
    log(f"#    half and scored {res['a_test']:.3f} held out; the pre-registered")
    log("#    trigger was 0.9, so this leg is a second reading rather than the")
    log("#    check on a positive. Same feature, same orientation, new images.")
    conf = run_continuous(n_nat=6, n_cart=3, seed=7, log=log)
    a_all = pooled_auc(vs(conf, "restart"), res["best"])
    a = a_all if res["sign"] > 0 else 1 - a_all
    log(f"      {res['best']} on the confirmation set: AUC {a:.3f} "
        f"(was {res['a_test']:.3f})")
    s = res["sign"]
    p, nf, nt = precision_at_full_recall(vs(conf, "restart"), res["best"], s)
    base = sum(w["opt"] for w in conf) / len(conf)
    log(f"      flagging every optimum there: precision {p:.3f} on {nf}/{nt} "
        f"flagged, against a base rate of {base:.3f}")
    log("")
    log(f"# checks failed: {FAIL if FAIL else 'none'}")
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
