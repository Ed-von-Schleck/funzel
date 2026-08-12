"""E6: can a wavelet transform replace the atom-placement search?

E5 settled what wavelets could plausibly be FOR. Its finding is that off-grid
refinement dominates dictionary choice: a structured dictionary wins clearly
on-grid, but after both sides are polished the advantage shrinks to a few
percent and at 64 splats the UNSTRUCTURED dictionary wins outright. So a
wavelet-shaped dictionary is not a promising route to a better optimum -- and
the closest thing already tested, the difference-of-Gaussians (a Mexican-hat
analogue), was the worst dictionary on-grid in both E4 and E5.

That leaves one role where wavelets could still matter, and it is about COST,
not quality. Every method in this repository places atoms by searching: greedy
evaluates the correlation of the residual against the whole dictionary at every
step, and SFW does the same inside its inner loop. That search is what section
9.3's certification ceiling and U5 are blocked on. A wavelet transform produces
a multiscale, oriented, position-resolved decomposition in O(n^2) with no search
at all, and thresholding it is provably near-optimal N-term selection for the
relevant smoothness classes. If it initialises Gaussians as well as the search
does, the search is unnecessary.

This is also what section 4's theorem actually does. Erb-Hangelbroek-Ron build
their N-term Gaussian approximant by taking the curvelet expansion, keeping its
largest coefficients, and replacing each retained curvelet by a small Gaussian
mixture. Section 4.1 dismisses this as "an achievability result for the
dictionary, not an encoder" because the sub-budgets are non-adaptive -- but the
BUDGETING is what is non-adaptive, while WHICH curvelets get approximated is
chosen from the data. Read as an algorithm rather than a proof, the theorem is
a transform-then-fit encoder that needs no search.

METHOD. Take a separable 2D DWT, rank coefficients by magnitude (valid as an
energy ranking for orthogonal families), keep the top ones under the splat
budget, and map each to one Gaussian: position from the coefficient's location,
scale from its level, orientation and aspect from its subband (horizontal
detail -> elongated along x, vertical -> along y, diagonal and approximation ->
isotropic). Then refit amplitudes by least squares and refine off-grid exactly
as E5 does, so the only thing that differs is where the atoms started.

Families include haar (the simplest hat-like step wavelet) and bior2.2 (a
linear-spline "hat" biorthogonal wavelet), since those were asked about
specifically, plus db4 and sym4 as smoother controls.

WHAT THIS CAN SHOW. Whether wavelet placement reaches the same refined error as
searched placement, and at what fraction of the placement cost.

WHAT THIS CANNOT SHOW. Separable DWTs have only three orientations per scale,
so a negative result here is evidence against SEPARABLE wavelets, not against
oriented systems in general -- curvelets and shearlets resolve orientation far
better and are what section 4's theorem actually uses. Stated up front (M5) so
a null result is not over-read.
"""

import sys
import time

import numpy as np
import pywt

from e2_relaxation_gap import _grid, atoms, render, fit_fixed_support, NP_ATOM
import e2b_natural as e2b
import e5_dictionary_scaling as e5


def wavelet_atoms(img, n, wavelet, budget, scale_k=0.9, aniso=2.2):
    """Top-`budget` DWT coefficients -> Gaussian splat parameters.

    One splat per retained coefficient, so the splat budget is the coefficient
    budget. No search: the transform is O(n^2) and the ranking is a sort."""
    L = int(np.log2(n)) - 1
    coeffs = pywt.wavedec2(img, wavelet, level=L, mode="periodization")
    cand = []                                  # (|val|, val, cx, cy, sig, orient)
    cA = coeffs[0]
    m = cA.shape[0]
    sig_A = n / m * scale_k
    for i in range(m):
        for j in range(m):
            cand.append((abs(cA[i, j]), cA[i, j], (j + .5) / m, (i + .5) / m,
                         sig_A, "iso"))
    for lev, (cH, cV, cD) in enumerate(coeffs[1:]):
        m = cH.shape[0]
        sig = n / m * scale_k
        for band, arr in (("h", cH), ("v", cV), ("d", cD)):
            idx = np.argsort(-np.abs(arr), axis=None)[:budget]
            for k in idx:
                i, j = divmod(int(k), m)
                cand.append((abs(arr[i, j]), arr[i, j],
                             (j + .5) / m, (i + .5) / m, sig, band))
    cand.sort(key=lambda t: -t[0])
    th, amp = [], []
    for (_, val, cx, cy, sig, orient) in cand[:budget]:
        s = max(sig, 1.0) / n                  # unit-square sigma, >= 1px
        if orient == "h":
            sx, sy = s * aniso, s / aniso
        elif orient == "v":
            sx, sy = s / aniso, s * aniso
        else:
            sx, sy = s, s
        th.append([cx, cy, np.log(1.0 / sx), 0.0, np.log(1.0 / sy)])
        amp.append(val)
    return np.array(th), np.array(amp, dtype=float)


def fit_from_init(th, y, X, Y, n, refine=True):
    """Least-squares amplitudes on the given atoms, then optional off-grid
    refinement -- identical to E5's, so only the initialisation differs."""
    lo, hi = e5.bounds(n)
    th = np.clip(th, lo, hi)
    G = atoms(th, X, Y)[0]
    c = np.linalg.lstsq(G.T, y, rcond=None)[0]
    r = y - c @ G
    e_init = 0.5 * float(r @ r)
    if not refine:
        return e_init, e_init
    _, _, f = fit_fixed_support(c, th, y, X, Y, 0.0, lo, hi, maxiter=900)
    return e_init, float(f)


def run(name, n=64, budgets=(8, 16, 32, 64),
        wavelets=("haar", "bior2.2", "db4", "sym4"), seed=0, log=print):
    X, Y = _grid(n)
    rng = np.random.default_rng(seed)
    y = e2b.target(name, n, rng)
    img = y.reshape(n, n)
    half = 0.5 * float(y @ y)
    pc = lambda e: 100.0 * e / half
    dicts = e5.make_dicts(n)
    log(f"# target={name} n={n} 0.5||y||^2={half:.3f}")
    log("")
    log(f"{'budget':>7} {'placement':>22} {'on init':>9} {'+refined':>9} "
        f"{'place s':>8}")
    rows = []
    for B in budgets:
        # searched placement, the incumbent: greedy over each dictionary
        for key in ("gauss unstructured", "gauss parabolic"):
            G, S = dicts[key]
            t0 = time.time()
            idx, c, eg, used = e5.greedy_splats(G, S, y, B)
            tsearch = time.time() - t0
            th, amp = e5.expand(S, idx, c)
            _, er = fit_from_init(th, y, X, Y, n)
            log(f"{B:7d} {'greedy/' + key.split()[1]:>22} {pc(eg):9.4f} "
                f"{pc(er):9.4f} {tsearch:8.2f}")
            rows.append(dict(budget=B, method="greedy/" + key, refined=er,
                             t=tsearch))
        # wavelet placement: transform + sort, no search
        for w in wavelets:
            t0 = time.time()
            th, amp = wavelet_atoms(img, n, w, B)
            tsearch = time.time() - t0
            ei, er = fit_from_init(th, y, X, Y, n)
            log(f"{B:7d} {'wavelet/' + w:>22} {pc(ei):9.4f} {pc(er):9.4f} "
                f"{tsearch:8.2f}")
            rows.append(dict(budget=B, method="wavelet/" + w, refined=er,
                             t=tsearch))
        # Control: random placement. Greedy and the wavelets are deterministic,
        # so a single draw would be an unfair comparison in either direction --
        # several are run and both the median and the best are reported.
        ers = []
        for s in range(5):
            rr = np.random.default_rng(1000 + s)
            th0 = np.column_stack([
                rr.uniform(0.05, 0.95, B), rr.uniform(0.05, 0.95, B),
                np.log(rr.uniform(2.0, 20.0, B)), np.zeros(B),
                np.log(rr.uniform(2.0, 20.0, B))])
            ers.append(fit_from_init(th0, y, X, Y, n)[1])
        er = float(np.median(ers))
        log(f"{B:7d} {'random x5 (control)':>22} {'':>9} {pc(er):9.4f} {0.0:8.2f}"
            f"   [best {pc(min(ers)):.4f}]")
        rows.append(dict(budget=B, method="random", refined=er, t=0.0))
        sub = [r for r in rows if r["budget"] == B]
        bw = min((r for r in sub if r["method"].startswith("wavelet")),
                 key=lambda r: r["refined"])
        bg = min((r for r in sub if r["method"].startswith("greedy")),
                 key=lambda r: r["refined"])
        log(f"{'':7} {'-> refined:':>22} best wavelet {bw['method'].split('/')[1]} "
            f"{pc(bw['refined']):.4f} vs best greedy {pc(bg['refined']):.4f} "
            f"({100*(bw['refined']-bg['refined'])/bg['refined']:+.1f}%), "
            f"placement {bg['t']/max(bw['t'],1e-9):.0f}x cheaper")
        log("")
    return rows


def main(targets=("cartoon", "ascent", "face"), n=64,
         budgets=(8, 16, 32, 64), out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E6: wavelet placement vs searched placement, both refined identically.")
    log(f"# params: n={n} budgets(splats)={budgets} targets={targets}")
    for t in targets:
        log("")
        log("=" * 80)
        try:
            run(t, n=n, budgets=budgets, log=log)
        except Exception:
            import traceback
            log(f"# {t} FAILED:")
            for ln in traceback.format_exc().splitlines():
                log("#   " + ln)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    tg = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else \
        ("cartoon", "ascent", "face")
    main(targets=tg, out=sys.argv[2] if len(sys.argv) > 2 else None)
