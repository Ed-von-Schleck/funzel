"""E7: frequency-domain fitting is a no-op; frequency CONTINUATION is not.

Two things get called "working in the frequency domain" and they behave
completely differently for this problem.

1. FITTING IN THE FOURIER DOMAIN IS EXACTLY THE SAME PROBLEM. By Parseval,
   ||Phi m - y||^2 in space equals (1/n^2)||F(Phi m) - F(y)||^2 in frequency,
   so the objective is identical up to a constant. The Gaussian dictionary is
   moreover closed under the Fourier transform -- the transform of a Gaussian
   with covariance Sigma is a Gaussian with covariance Sigma^{-1} -- so the
   dictionary maps to a dictionary of the same family. Parseval also preserves
   every inner product, so the Gram matrix, the coherence, the local minima and
   the global optimum are all literally unchanged. Re-deriving the method in
   frequency cannot alter the optimisation landscape by even a little. It is
   not a foolish idea, it is a provably empty one.

   Two things do change, and both are already in play. The transform is a
   computational device -- eta_sup already searches positions with an FFT
   correlation. And a frequency-WEIGHTED L2 is a different objective while
   remaining Hilbertian, so it keeps the adjoint that section 6's certificate
   needs; that is a live option for U9 (perceptual acceptability) which plain
   L2 cannot address and SSIM cannot certify.

2. FREQUENCY CONTINUATION IS A DIFFERENT ALGORITHM, NOT A DIFFERENT PROBLEM,
   and it attacks local minima directly. Low-pass the target hard, fit, then
   sharpen progressively, carrying the solution forward. A blurred target has a
   smoother, flatter objective with fewer local minima; tracking the minimiser
   as the target sharpens is graduated non-convexity, a classical route to a
   better optimum on non-convex problems. This is the only version of the idea
   that can change which optimum is reached, and it is roughly what FreGS does
   for 3D Gaussian splatting (progressive low-to-high frequency guidance),
   where it is reported to help. Nothing in this repository has tested it.

FAIRNESS. Continuation could win simply by getting more optimiser iterations,
so every method here is given the SAME total L-BFGS budget: a direct fit gets
all of it on the true target, a K-stage continuation gets total/K per stage.
Any difference is the schedule, not the compute.

WHAT THIS CAN SHOW. Whether continuation reaches a better optimum than direct
refinement from the same initialisation, at equal budget and equal compute.

WHAT THIS CANNOT SHOW. It compares reachable optima, not distance to the true
optimum, which is unknown here (E4 is the only place one was computed, at
N<=4). Three targets, 64^2, one continuation schedule.
"""

import sys
import time

import numpy as np

from e2_relaxation_gap import _grid, atoms, fit_fixed_support
import e2b_natural as e2b
import e5_dictionary_scaling as e5


def lowpass(y, n, sigma_px):
    """Isotropic Gaussian low-pass, applied as a multiplier in Fourier."""
    if sigma_px <= 0:
        return y.copy()
    img = y.reshape(n, n)
    fx = np.fft.fftfreq(n) * n
    FX, FY = np.meshgrid(fx, fx, indexing="ij")
    H = np.exp(-2.0 * (np.pi ** 2) * (sigma_px ** 2) * ((FX / n) ** 2 + (FY / n) ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H)).ravel()


def continuation(th0, amp0, y, X, Y, n, schedule, total_iter=900,
                 handicap=False):
    """Fit through a low-pass schedule, carrying the solution forward.

    handicap=False splits total_iter across the stages, so continuation and
    direct get equal compute -- but then continuation's FINAL stage, the only
    one run on the true target, gets total_iter/K while direct gets all of it.
    That confound could sink continuation on its own. handicap=True instead
    gives the final stage the full total_iter and treats the coarse stages as
    extra, so continuation is strictly advantaged; a loss there is decisive."""
    lo, hi = e5.bounds(n)
    th, amp = np.clip(th0, lo, hi), amp0.copy()
    coarse = [s for s in schedule if s > 0]
    per = (total_iter // max(1, len(schedule))) if not handicap else \
        max(50, total_iter // max(1, len(coarse)))
    for s in coarse:
        ys = lowpass(y, n, s)
        amp, th, _ = fit_fixed_support(amp, th, ys, X, Y, 0.0, lo, hi, maxiter=per)
    amp, th, f = fit_fixed_support(amp, th, y, X, Y, 0.0, lo, hi,
                                   maxiter=total_iter if handicap else per)
    return float(f)


def direct(th0, amp0, y, X, Y, n, total_iter=900):
    lo, hi = e5.bounds(n)
    _, _, f = fit_fixed_support(amp0, np.clip(th0, lo, hi), y, X, Y, 0.0,
                                lo, hi, maxiter=total_iter)
    return float(f)


def inits(y, X, Y, n, B, dicts, rng):
    """Initialisations to compare: greedy placement, and random placement."""
    out = {}
    G, S = dicts["gauss unstructured"]
    idx, c, _, _ = e5.greedy_splats(G, S, y, B)
    out["greedy"] = e5.expand(S, idx, c)
    th0 = np.column_stack([
        rng.uniform(0.05, 0.95, B), rng.uniform(0.05, 0.95, B),
        np.log(rng.uniform(2.0, 20.0, B)), np.zeros(B),
        np.log(rng.uniform(2.0, 20.0, B))])
    Gr = atoms(th0, X, Y)[0]
    out["random"] = (th0, np.linalg.lstsq(Gr.T, y, rcond=None)[0])
    return out


def run(name, n=64, budgets=(8, 16, 32, 64), total_iter=900, n_seeds=3,
        log=print):
    X, Y = _grid(n)
    y = e2b.target(name, n, np.random.default_rng(0))
    half = 0.5 * float(y @ y)
    pc = lambda e: 100.0 * e / half
    dicts = e5.make_dicts(n)
    sched = [8.0, 4.0, 2.0, 1.0, 0.0]
    log(f"# target={name} n={n} 0.5||y||^2={half:.3f}")
    log(f"# schedule sigma(px) = {sched}; every method gets {total_iter} "
        f"L-BFGS iterations total")
    log("")
    log(f"{'budget':>7} {'init':>8} {'direct':>9} {'cont(equal)':>12} "
        f"{'change':>8} {'cont(handicap)':>15} {'change':>8}")
    rows = []
    for B in budgets:
        for key in ("greedy", "random"):
            ds, cs, hs = [], [], []
            for s in range(n_seeds if key == "random" else 1):
                rng = np.random.default_rng(2000 + s)
                th0, amp0 = inits(y, X, Y, n, B, dicts, rng)[key]
                ds.append(direct(th0, amp0, y, X, Y, n, total_iter))
                cs.append(continuation(th0, amp0, y, X, Y, n, sched, total_iter))
                hs.append(continuation(th0, amp0, y, X, Y, n, sched, total_iter,
                                       handicap=True))
            d, c, h = (float(np.median(ds)), float(np.median(cs)),
                       float(np.median(hs)))
            log(f"{B:7d} {key:>8} {pc(d):9.4f} {pc(c):12.4f} "
                f"{100*(c-d)/d:+7.2f}% {pc(h):15.4f} {100*(h-d)/d:+7.2f}%")
            rows.append(dict(budget=B, init=key, direct=d, cont=c, handi=h))
    log("")
    log("  change < 0 means continuation reached a better optimum. 'equal' splits")
    log("  the iteration budget across stages; 'handicap' gives the final stage")
    log("  the FULL budget and the coarse stages on top, so continuation is")
    log("  strictly advantaged and a loss there cannot be a compute artifact.")
    return rows


def main(targets=("cartoon", "ascent", "face"), n=64,
         budgets=(8, 16, 32, 64), out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E7: frequency continuation vs direct refinement, equal compute.")
    log("# Plain Fourier-domain fitting is omitted: by Parseval it is the same")
    log("# problem, so there is nothing to measure.")
    for t in targets:
        log("")
        log("=" * 70)
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
