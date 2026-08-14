"""E18 (U31): is the restart floor a convergence artefact, and does it fall?

Section 10.16 found the binding constraint on a canonical encoder. It is not the
placement rule -- a thirtyfold more stable initialisation buys 18% in the
converged encoding -- it is the optimiser. Re-running Adam on an UNCHANGED image
from its own converged solution moves the atoms 1.5 to 3.2 px, which is the same
order as the entire response to a 30dB perturbation. Nothing upstream can beat
that floor.

Section 10.15 also showed why to suspect it: re-running improves PSNR every time,
by 0.28 to 0.46 dB. Four thousand Adam steps is not convergence, so the floor may
simply be unfinished optimisation. If so it should fall with more steps, and the
perturbation response should fall with it.

THREE SETTINGS, all on the same deterministic quadtree encoder:
  adam4000    the incumbent, and the setting every earlier stability number in
              this repository was measured at.
  adam16000   four times the steps. If the floor is unfinished optimisation this
              is where it starts to show.
  adam+lbfgs  4000 Adam steps then L-BFGS to a tight tolerance. The interesting
              one: L-BFGS is deterministic and carries NO momentum state, so
              restarting it at its own output starts from the same place with
              the same state. If it has actually reached a stationary point the
              restart cannot move at all and the floor is exactly zero. If it
              moves, the point was not stationary.

THREE MEASUREMENTS PER SETTING, per image:
  floor       re-run the same optimiser from its own output on the SAME image.
              This is the reproducibility of the optimiser, with the image held
              fixed, and it is the quantity section 10.16 identified as binding.
  noise       encode the 30dB-perturbed image from scratch and compare. This is
              what a canonical encoder actually needs to be small.
  quality     PSNR, so that a setting cannot buy stability by underfitting.

Displacement is reported as median AND max over atoms. Section 10.16 found the
median alone reading 0.00 px where the max was 34.93 -- half the atoms unmoved
while one crossed half the image -- and that nearly produced a false conclusion.

WHAT THIS CANNOT SHOW. 64x64 images, three of them, N=36, one initialisation.
The floor falling would not by itself make the encoder canonical: the
perturbation response has to fall with it, which is why both are measured. And a
floor of zero from L-BFGS would mean the optimiser is reproducible, not that the
solution is good -- L-BFGS stops at whatever stationary point it reaches, and
section 10.10 part C found it weaker than Adam on this problem.
"""

import sys
import time

import numpy as np

from e2_relaxation_gap import _grid, atoms, loss_grad, fit_fixed_support
import e5_dictionary_scaling as e5
import e13_hit_rate as e13
import e14_certifiable as e14
import e15_canonical_quality as e15
import e16_quantised_stability as e16

FAIL = []


def check(name, ok, detail=""):
    print(f"    [{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def lbfgs(th0, y, X, Y, n, maxiter=3000):
    """L-BFGS from a given start, amplitudes refit first. No momentum state."""
    lo, hi = e5.bounds(n)
    th0 = np.clip(th0, lo, hi)
    G = atoms(th0, X, Y)[0]
    c0 = np.linalg.lstsq(G.T, y, rcond=None)[0]
    _, th, _ = fit_fixed_support(c0, th0, y, X, Y, 0.0, lo, hi, maxiter=maxiter)
    return th


def encode(y, N, n, X, Y, setting, th0=None):
    """One encoder run. th0 given means a restart from an existing solution."""
    if th0 is None:
        th0 = e15.init_structure(y, N, n)
    if setting == "adam4000":
        return e15.adam(th0, y, X, Y, n, checkpoints=(4000,))[4000][1]
    if setting == "adam16000":
        return e15.adam(th0, y, X, Y, n, checkpoints=(16000,))[16000][1]
    if setting == "adam+lbfgs":
        th = e15.adam(th0, y, X, Y, n, checkpoints=(4000,))[4000][1]
        return lbfgs(th, y, X, Y, n)
    raise ValueError(setting)


def grad_norm(th, y, X, Y, n):
    """Scale-free measure of how stationary a solution is.

    Reported as the gradient's infinity norm against the atom parameters,
    divided by the loss, so it is comparable across images and budgets. A
    genuinely converged solution has this near zero."""
    G = atoms(th, X, Y)[0]
    c = np.linalg.lstsq(G.T, y, rcond=None)[0]
    z = np.concatenate([c, th.ravel()])
    f, g = loss_grad(z, y, X, Y, 0.0)
    return float(np.abs(g[len(c):]).max() / max(f, 1e-300))


SETTINGS = ("adam4000", "adam16000", "adam+lbfgs")


def run(imgs, n, N, log=print):
    X, Y = _grid(n)
    rng = np.random.default_rng(11)
    log(f"  N={N}, quadtree init, three optimiser settings. Displacement is "
        f"median / max over atoms, in pixels.")
    log(f"      {'setting':>11} {'PSNR':>7} {'dPSNR':>7} {'|grad|/loss':>12} "
        f"{'restart floor':>16} {'30dB noise':>16}")
    log(f"      {'':>11} {'':>7} {'restart':>7} {'':>12} "
        f"{'(same image)':>16} {'(new encode)':>16}")
    rows = []
    for s in SETTINGS:
        t0 = time.time()
        acc = {"floor": [], "noise": []}
        ps, dps, gn = [], [], []
        for (nm, y) in imgs:
            th = encode(y, N, n, X, Y, s)
            ps.append(e16.fit_psnr(th, y, X, Y, n))
            gn.append(grad_norm(th, y, X, Y, n))
            again = encode(y, N, n, X, Y, s, th0=th)
            acc["floor"].append(e13.matched(th, again, n))
            dps.append(e16.fit_psnr(again, y, X, Y, n) - ps[-1])
            y1 = e16.perturbations(y, n, rng)[0][1]          # 30dB noise
            acc["noise"].append(e13.matched(th, encode(y1, N, n, X, Y, s), n))
        cells = []
        for k in ("floor", "noise"):
            a = np.array(acc[k])
            cells.append(f"{np.median(a[:, 0]):6.2f} /{np.median(a[:, 1]):6.2f}")
        log(f"      {s:>11} {np.mean(ps):7.3f} {np.mean(dps):+7.3f} "
            f"{np.median(gn):12.2e} " + " ".join(f"{c:>16}" for c in cells)
            + f"   [{time.time() - t0:.0f}s]")
        rows.append(dict(setting=s, psnr=float(np.mean(ps)),
                         dpsnr=float(np.mean(dps)), grad=float(np.median(gn)),
                         **{k: [float(np.median(np.array(v)[:, 0])),
                                float(np.median(np.array(v)[:, 1]))]
                            for k, v in acc.items()}))
    return rows


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    n, N = 64, 36
    imgs = e14.image_set(n, np.random.default_rng(5), n_nat=2, n_cart=1, log=log)
    log("# E18 (U31): is the optimiser's restart floor just unfinished")
    log("# optimisation? Section 10.16 showed it binds: no placement rule can")
    log("# beat it, and it is the same size as the whole perturbation response.")
    log("")

    X, Y = _grid(n)
    y0 = imgs[0][1]
    th = e15.init_structure(y0, N, n)
    # An earlier version of this check ran L-BFGS at maxiter=200 and asked
    # whether restarting moved less than Adam's 1.47px. It moved 2.29px and the
    # check "failed" -- but a truncated L-BFGS restarts in mid-descent and is
    # supposed to keep moving, so the check was testing the wrong object, and
    # the displacement it wanted is the floor column of the table below anyway.
    # Replaced by an invariant that cannot be satisfied by accident.
    G = atoms(np.clip(th, *e5.bounds(n)), X, Y)[0]
    c0 = np.linalg.lstsq(G.T, y0, rcond=None)[0]
    f0 = 0.5 * float(((c0 @ G - y0) ** 2).sum())
    p = lbfgs(th, y0, X, Y, n, maxiter=200)
    Gp = atoms(p, X, Y)[0]
    cp = np.linalg.lstsq(Gp.T, y0, rcond=None)[0]
    f1 = 0.5 * float(((cp @ Gp - y0) ** 2).sum())
    check("C1 L-BFGS decreases the loss it is given", f1 <= f0,
          f"({f0:.4f} -> {f1:.4f})")
    check("C2 more Adam steps do not make the fit worse",
          e16.fit_psnr(e15.adam(th, y0, X, Y, n, checkpoints=(4000, 8000))[8000][1],
                       y0, X, Y, n)
          >= e16.fit_psnr(e15.adam(th, y0, X, Y, n,
                                   checkpoints=(4000,))[4000][1], y0, X, Y, n))
    g_init = grad_norm(th, y0, X, Y, n)
    g_fit = grad_norm(lbfgs(th, y0, X, Y, n, maxiter=1000), y0, X, Y, n)
    check("C3 the stationarity measure falls when the solution is optimised",
          g_fit < g_init, f"({g_init:.2e} at init -> {g_fit:.2e} after L-BFGS)")

    log("")
    log("## Floor, perturbation response and quality against optimiser effort")
    run(imgs, n, N, log=log)

    log("")
    log(f"# checks failed: {FAIL if FAIL else 'none'}")
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
