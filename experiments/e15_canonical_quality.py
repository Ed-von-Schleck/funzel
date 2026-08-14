"""E15: can a CANONICAL encoder match the standard one on quality?

Sections 10.10-10.13 established what the canonical object is worth and what it
costs. The optimum is unique and stable, and it is unreachable: no certificate
(10.8, 10.9), no agreement signal (10.12), no solution-intrinsic tell (10.13).

That closes "know you have the global optimum". It does not close the thing the
global optimum was wanted FOR. The reason to want it was that it is a function
of the image alone, hence reproducible and comparable across images -- and that
property needs only DETERMINISM plus STABILITY, not optimality. A deterministic
encoder returns the same atoms for the same image by construction, and section
10.10 part B showed the grid optimum also survives noise, a one-pixel shift and
an intensity rescale.

So the question this asks is the one that decides whether any of it is useful:
IS A DETERMINISTIC ENCODER MEANINGFULLY WORSE THAN THE STANDARD RANDOM ONE? A
canonical representation that costs 30% of the reconstruction quality is a
curiosity. One that costs nothing is a free upgrade.

THE BASELINE IS THE RECIPE EVERYONE ACTUALLY USES: uniform-random placement
followed by Adam on every parameter, which is GaussianImage's method minus
densification and pruning. It is reimplemented here rather than taken from a
paper, so the comparison is internal -- same images, same budget, same optimiser,
same step count -- and no claim is made about matching published numbers.

THREE INITIALISATIONS, of which two are deterministic:
  random     uniform centres. The baseline. Run at two seeds, which also
             measures how much the standard recipe moves between runs.
  lattice    a regular grid of identical atoms. Deterministic and completely
             image-independent -- a NULL CONTROL for the whole idea. If this
             matches the baseline, initialisation does not matter at this budget
             and canonicity is free for the taking.
  structure  deterministic and image-adaptive: a binary quadtree split on cell
             variance until there are N cells, one atom per cell, oriented along
             the edge direction from the cell's structure tensor.

A FOURTH INITIALISATION IS A CONTROL, not an idea. The random and lattice inits
put atoms at sigma ~ 1/sqrt(N), but the quadtree's cells concentrate where the
image has detail, so its atoms start about five times smaller. Small atoms and
adaptive placement would otherwise be the same measurement. 'lattice-fine' is
the plain lattice at the quadtree's atom size, which separates them: if it
matches structure, the win is atom size; if it matches lattice, the win is
placement. Section 10.5 was confounded in exactly this way once, by a scale cap
that differed between the arms being compared.

QUALITY IS REPORTED AS PSNR, at two step counts, 1000 and 4000. The second
number matters as much as the first: if a difference at 1000 steps is gone by
4000, the initialisation was buying convergence speed rather than quality, and
the canonical encoder costs nothing at convergence.

STABILITY IS MEASURED, NOT ASSUMED. Determinism gives reproducibility on
identical input, which is trivial. What a canonical representation needs is that
a small change in the image produces a small change in the atoms -- otherwise
two similar images are not comparable and the whole point is lost. Each encoder
is re-run on the image perturbed by 30dB noise, by a one-pixel shift (undone
before matching), and by a 1% intensity rescale, and the matched atom
displacement is reported.

WHAT THIS CANNOT SHOW. 64x64 images at 36 and 144 splats, one optimiser, one
step budget, three images, two seeds for the stochastic baseline. The atoms-per-
pixel density at 144 splats matches the published regime but the absolute scale
does not: real results are at 10^3-10^4 primitives on far larger images and with
densification and pruning, neither of which is implemented here. A null result
at this scale does not establish one at that scale.

Greedy placement is not included: section 10.10 part C already found it no
better than random once Adam runs, and placing 144 atoms without duplication
needs a dictionary far larger than the budget.
"""

import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from e2_relaxation_gap import _grid, atoms, loss_grad, NP_ATOM
import e5_dictionary_scaling as e5
import e14_certifiable as e14

FAIL = []


def check(name, ok, detail=""):
    print(f"    [{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------- parametrisation
def theta_from_axes(cx, cy, angle, sL, sW):
    """Atom parameters for a Gaussian with axes (sL, sW) rotated by `angle`.

    The internal parametrisation is theta = [cx, cy, log M00, M01, log M11]
    with M the upper Cholesky factor of the INVERSE covariance, so this inverts
    that: build the covariance, invert, factor. Same construction as
    e4.build_parabolic, which is the only other place it appears."""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    A = np.linalg.inv(R @ np.diag([sL ** 2, sW ** 2]) @ R.T)
    M = np.linalg.cholesky(A).T
    return [cx, cy, np.log(M[0, 0]), M[0, 1], np.log(M[1, 1])]


def psnr(err, P):
    """err = 0.5||r||^2 on a [0,1]-scaled image with P pixels."""
    mse = 2.0 * err / P
    return 10.0 * np.log10(1.0 / max(mse, 1e-300))


# ------------------------------------------------------------ initialisations
def init_random(y, N, n, rng):
    """The standard recipe: uniform centres, isotropic-ish random scales."""
    s = 1.0 / np.sqrt(N)                  # same atom size as the plain lattice
    th = []
    for i in range(N):
        cx, cy = rng.uniform(0.05, 0.95, 2)
        sL, sW = s * rng.uniform(0.5, 2.0, 2)
        th.append(theta_from_axes(cx, cy, rng.uniform(0, np.pi), sL, sW))
    return np.array(th)


def _lattice(N, scale):
    k = int(np.ceil(np.sqrt(N)))
    s = scale / np.sqrt(N)
    ax = (np.arange(k) + 0.5) / k
    th = []
    for i in range(k):
        for j in range(k):
            if len(th) < N:
                th.append(theta_from_axes(ax[j], ax[i], 0.0, s, s))
    return np.array(th)


def init_lattice(y, N, n):
    """Deterministic and image-INDEPENDENT: the null control."""
    return _lattice(N, 1.0)


def init_lattice_fine(y, N, n):
    """The same lattice at the structure init's atom size.

    The quadtree concentrates cells where the image has detail, so its atoms
    start about five times smaller than a 1/sqrt(N) lattice: median 2.07px
    against 10.67px at N=36, and 1.01 against 5.33 at N=144, a ratio of 0.19 at
    both budgets. Without this control, 'adaptive placement wins' and 'small
    atoms win' are the same measurement."""
    return _lattice(N, 0.19)


def init_structure(y, N, n):
    """Deterministic and image-adaptive: variance quadtree + structure tensor.

    Binary splits along the longer side, so the cell count lands exactly on N
    rather than on 1+3k as a four-way split would."""
    img = y.reshape(n, n)
    gy, gx = np.gradient(img)
    cells = [(0, 0, n, n)]

    def score(c):
        i0, j0, h, w = c
        blk = img[i0:i0 + h, j0:j0 + w]
        return blk.var() * blk.size if min(h, w) >= 2 else -np.inf

    while len(cells) < N:
        sc = [score(c) for c in cells]
        if not np.isfinite(max(sc)):
            break                              # nothing left that can be split
        i0, j0, h, w = cells.pop(int(np.argmax(sc)))
        if h >= w:
            cells += [(i0, j0, h // 2, w), (i0 + h // 2, j0, h - h // 2, w)]
        else:
            cells += [(i0, j0, h, w // 2), (i0, j0 + w // 2, h, w - w // 2)]
    # deterministic order, so the encoding does not depend on split history
    cells.sort()
    th = []
    for (i0, j0, h, w) in cells[:N]:
        cy, cx = (i0 + h / 2) / n, (j0 + w / 2) / n
        bx, by = gx[i0:i0 + h, j0:j0 + w], gy[i0:i0 + h, j0:j0 + w]
        J = np.array([[float((bx * bx).sum()), float((bx * by).sum())],
                      [float((bx * by).sum()), float((by * by).sum())]])
        ev, evec = np.linalg.eigh(J)           # ascending
        # elongate ALONG the edge = the direction of least gradient variation
        ang = np.arctan2(evec[1, 0], evec[0, 0])
        aniso = 1.0 + 3.0 * (ev[1] - ev[0]) / max(ev[1] + ev[0], 1e-12)
        base = 0.5 * max(h, w) / n
        th.append(theta_from_axes(cx, cy, ang, base, base / aniso))
    while len(th) < N:                          # image too flat to split further
        th.append(th[-1])
    return np.array(th)


INITS = {"random": init_random, "lattice": init_lattice,
         "lattice-fine": init_lattice_fine, "structure": init_structure}

# stability is measured on the three distinct ideas, not on the scale control
STAB = ["random", "lattice", "structure"]


# ------------------------------------------------------------------ optimiser
def adam(th0, y, X, Y, n, checkpoints=(1000, 4000), lr=0.01):
    """Adam on amplitudes and atom parameters, recording at each checkpoint.

    One parameter, not two. An earlier version took both a step count and a
    checkpoint tuple; callers passed the checkpoints positionally into the step
    count, so the recording silently used the DEFAULT checkpoints and the run
    died much later, in aggregation, on a missing key."""
    lo, hi = e5.bounds(n)
    K = len(th0)
    th0 = np.clip(th0, lo, hi)
    A = atoms(th0, X, Y)[0]
    c0 = np.linalg.lstsq(A.T, y, rcond=None)[0]
    z = np.concatenate([c0, th0.ravel()])
    zlo = np.concatenate([np.full(K, -np.inf), np.tile(lo, K)])
    zhi = np.concatenate([np.full(K, np.inf), np.tile(hi, K)])
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    b1, b2, eps = 0.9, 0.999, 1e-8
    out = {}
    for t in range(1, max(checkpoints) + 1):
        f, g = loss_grad(z, y, X, Y, 0.0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        z = np.clip(z - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t))
                                                    + eps), zlo, zhi)
        if t in checkpoints:
            f, _ = loss_grad(z, y, X, Y, 0.0)
            out[t] = (float(f), z[K:].reshape(K, NP_ATOM).copy())
    assert set(out) == set(checkpoints), (sorted(out), sorted(checkpoints))
    return out


def matched_px(thA, thB, n):
    """Median matched centre distance in px, N! symmetry quotiented out."""
    A, B = thA[:, :2] * n, thB[:, :2] * n
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    r, c = linear_sum_assignment(D)
    return float(np.median(D[r, c]))


# ------------------------------------------------------------------ quality
def run_quality(imgs, n, budgets, steps, seeds=2, log=print):
    X, Y = _grid(n)
    P = n * n
    log(f"  {n}x{n} images, Adam lr=0.01, PSNR at {steps[0]} and {steps[-1]} "
        f"steps; 'random' is the standard recipe at {seeds} seeds")
    log(f"      {'budget':>6} {'method':>10} {'PSNR@' + str(steps[0]):>11} "
        f"{'PSNR@' + str(steps[-1]):>11} {'vs random':>10} {'seed spread':>12}")
    rows = []
    for N in budgets:
        per = {}
        for name in INITS:
            got = []
            for s in range(seeds if name == "random" else 1):
                for (nm, y) in imgs:
                    rng = np.random.default_rng(600 + 17 * s)
                    th0 = (INITS[name](y, N, n, rng) if name == "random"
                           else INITS[name](y, N, n))
                    r = adam(th0, y, X, Y, n, steps)
                    got.append({k: (psnr(v[0], P), v[1]) for k, v in r.items()})
            per[name] = got
        base = np.mean([g[steps[-1]][0] for g in per["random"]])
        for name in INITS:
            g = per[name]
            p1 = np.mean([q[steps[0]][0] for q in g])
            p2 = np.mean([q[steps[-1]][0] for q in g])
            if name == "random" and seeds > 1:
                per_img = np.array([q[steps[-1]][0] for q in g]).reshape(
                    seeds, len(imgs))
                spread = f"{per_img.std(axis=0).mean():.3f} dB"
            else:
                spread = "0 (determ.)"
            log(f"      {N:6d} {name:>10} {p1:11.3f} {p2:11.3f} "
                f"{p2 - base:+10.3f} {spread:>12}")
            rows.append(dict(N=N, method=name, psnr1=p1, psnr2=p2,
                             delta=p2 - base))
    return rows


# ---------------------------------------------------------------- stability
def run_stability(imgs, n, N, steps, log=print):
    X, Y = _grid(n)
    P = n * n
    log(f"  matched atom displacement under perturbation, N={N}, "
        f"{steps[-1]} Adam steps")
    log(f"      {'method':>10} {'perturbation':>14} {'displacement':>13} "
        f"{'dPSNR':>8}")
    rng0 = np.random.default_rng(7)
    rows = []
    for name in STAB:
        acc = {}
        for (nm, y) in imgs:
            th0 = (INITS[name](y, N, n, np.random.default_rng(600))
                   if name == "random" else INITS[name](y, N, n))
            ref = adam(th0, y, X, Y, n, steps)[steps[-1]]
            sig = np.sqrt((y @ y) / len(y) * 1e-3)          # 30 dB
            for label, y1, shift in [
                ("noise 30dB", y + rng0.normal(0, sig, len(y)), (0, 0)),
                ("shift 1px", np.roll(y.reshape(n, n), 1, axis=1).ravel(),
                 (1, 0)),
                ("scale x1.01", y * 1.01, (0, 0)),
            ]:
                th1 = (INITS[name](y1, N, n, np.random.default_rng(600))
                       if name == "random" else INITS[name](y1, N, n))
                e1, thp = adam(th1, y1, X, Y, n, steps)[steps[-1]]
                thc = thp.copy()
                thc[:, 0] -= shift[0] / n        # undo the known shift first
                d = matched_px(ref[1], thc, n)
                acc.setdefault(label, []).append(
                    (d, psnr(e1, P) - psnr(ref[0], P)))
        for label, vals in acc.items():
            d = float(np.median([v[0] for v in vals]))
            dp = float(np.median([v[1] for v in vals]))
            log(f"      {name:>10} {label:>14} {d:10.2f} px {dp:+8.3f}")
            rows.append(dict(method=name, perturbation=label, disp=d, dpsnr=dp))
    return rows


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    # 64x64 at 144 splats is one atom per 28 pixels, which is the density
    # published splatting results work at; the budgets are perfect squares so
    # the lattice control gets a full grid rather than a partial last row.
    n, budgets, steps = 64, (36, 144), (1000, 4000)
    rng = np.random.default_rng(5)
    imgs = e14.image_set(n, rng, n_nat=2, n_cart=1, log=log)
    X, Y = _grid(n)

    log("# E15: is a canonical (deterministic) encoder worse than the standard")
    log("# random-init recipe? Quality first, then stability.")
    log("")
    # instrument checks before interpretation
    for name, fn in INITS.items():
        th = (fn(imgs[0][1], 36, n, np.random.default_rng(0)) if name == "random"
              else fn(imgs[0][1], 36, n))
        check(f"C1 {name} init returns 36 finite atoms",
              th.shape == (36, NP_ATOM) and np.isfinite(th).all())
    y0 = imgs[0][1]
    th = init_structure(y0, 36, n)
    check("C2 structure init is deterministic",
          np.array_equal(th, init_structure(y0, 36, n)))
    thr = init_random(y0, 36, n, np.random.default_rng(0))
    check("C3 random init is NOT deterministic across seeds",
          not np.array_equal(thr, init_random(y0, 36, n,
                                              np.random.default_rng(1))))
    A = atoms(np.array([theta_from_axes(0.5, 0.5, 0.0, 0.1, 0.05)]), X, Y)[0][0]
    im = A.reshape(n, n)
    ci, cj = np.unravel_index(int(np.argmax(im)), im.shape)
    sx = np.sqrt((im.sum(axis=0) * ((np.arange(n) + 0.5) / n - 0.5) ** 2).sum()
                 / im.sum(axis=0).sum())
    sy = np.sqrt((im.sum(axis=1) * ((np.arange(n) + 0.5) / n - 0.5) ** 2).sum()
                 / im.sum(axis=1).sum())
    check("C4 theta_from_axes reproduces the requested axes",
          abs(sx - 0.1) < 0.005 and abs(sy - 0.05) < 0.005
          and abs(ci - n // 2) <= 1 and abs(cj - n // 2) <= 1,
          f"(asked 0.100/0.050, measured {sx:.3f}/{sy:.3f}, "
          f"peak at {cj},{ci} of {n//2})")

    log("")
    log("## A. Quality against the standard recipe, at matched splat budget")
    t0 = time.time()
    run_quality(imgs, n, budgets, steps, log=log)
    log(f"  [{time.time() - t0:.0f}s]")

    log("")
    log("## B. Stability: does a small change in the image move the atoms?")
    t0 = time.time()
    run_stability(imgs[:2], n, budgets[0], steps, log=log)
    log(f"  [{time.time() - t0:.0f}s]")

    log("")
    log(f"# checks failed: {FAIL if FAIL else 'none'}")
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
