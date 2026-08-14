"""E17 (U30): does a continuously varying placement rule fix the instability?

Section 10.15 located the problem. A deterministic encoder still moves its atoms
2.7-5.0 px when the image is perturbed imperceptibly, but warm-starting the
perturbed image from the clean image's converged solution lands at 1.90 px
against a 1.47 px floor for simply restarting Adam. So the optimum does not
move. The INITIALISATION does, and the optimiser then converges somewhere else.

The initialisation is a variance quadtree, and its decisions are discrete: at
each step it takes the argmax over a list of cells and splits it. Two images
that differ by nothing visible can order that list differently, and a flip near
the top changes every split that follows. There is no reason for such a rule to
be stable, and section 10.15 measured that it is not.

The fix this tests is to remove the discrete decisions entirely. Two placement
rules, both deterministic and both continuous functions of the image:

  softgrid   a fixed grid of Gaussian windows; each atom sits at the
             density-weighted centroid of its own window. Nothing is chosen --
             every atom is a weighted average -- so the map from image to atoms
             is as continuous as the density is. Barely adaptive, and that is
             the point: it is the maximally stable end of the range.
  lloyd      Lloyd's algorithm on the same density: hard Voronoi cells,
             density-weighted centroids, iterated. Adaptive. It does contain an
             argmin, and the point of including it is that not every argmin is
             destabilising: when a Voronoi boundary flips, a sliver of area
             changes cells and the centroids move infinitesimally, whereas when
             the quadtree's argmax flips, a different cell is split and every
             later split changes.

The fully continuous iterated version -- soft assignment, run to convergence --
was written and then dropped. Coincident atoms are a fixed point of any soft
update, so the atoms merge: 0.08px between the closest pair at N=144 against a
5.33px lattice spacing. Full continuity is not automatically the right target;
what is needed is that small input changes produce small output changes, and an
algorithm that destroys its own budget does not qualify.

Both take their density from a blurred gradient magnitude, their orientation and
elongation from a blurred structure tensor sampled bilinearly, and their size
from the mean distance to the three nearest atoms. Every one of those is a
continuous function of the image. Sizes are calibrated to the quadtree's median
atom size, because section 10.14 found initial atom size to be the single
largest effect on quality and an uncontrolled difference there would swamp
everything else.

PART A measures the INITIALISATIONS alone, before any optimisation. This is the
direct test of the mechanism and it costs nothing to run: if the quadtree's init
moves far under perturbation and these do not, the diagnosis in 10.15 is right.

PART B measures the converged encodings, which is what actually matters. A
continuous init is only useful if the stability survives 4000 Adam steps. Read
against the 1.47 px floor from section 10.15, not against zero.

PART C checks quality did not regress. A stable encoder that reconstructs badly
is not an improvement, and section 10.14 set the bar: a scale-tuned random
baseline.

WHAT THIS CANNOT SHOW. 64x64 images, three of them, N=36 and 144, one optimiser,
4000 steps -- which section 10.15 showed is not convergence. The floor it is
read against was measured on the quadtree encoder; if a different init has a
different restart floor, part B needs its own, so part B measures one per arm.
"""

import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from e2_relaxation_gap import _grid
import e14_certifiable as e14
import e15_canonical_quality as e15
import e16_quantised_stability as e16
import e13_hit_rate as e13

FAIL = []


def check(name, ok, detail=""):
    print(f"    [{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------- fields
def fields(y, n, blur=2.0, eps=0.02, gamma=0.7):
    """Density and structure tensor, both blurred so both vary smoothly.

    The blur is what makes these continuous in practice: an unblurred gradient
    field changes abruptly under noise, and every quantity below is an average
    against it."""
    img = y.reshape(n, n)
    gy, gx = np.gradient(img)
    rho = gaussian_filter(np.hypot(gx, gy), blur)
    rho = (rho + eps) ** gamma
    rho = rho / rho.sum()
    J = [gaussian_filter(a, blur) for a in (gx * gx, gx * gy, gy * gy)]
    return rho, J


def sample(F, px, py, n):
    """Bilinear sample of a field at unit-square coordinates.

    Bilinear rather than nearest: nearest-pixel lookup is a step function of
    position, which would put a discrete decision back into a rule whose whole
    purpose is not to have one."""
    x = np.clip(px * n - 0.5, 0, n - 1.001)
    yq = np.clip(py * n - 0.5, 0, n - 1.001)
    j, i = np.floor(x).astype(int), np.floor(yq).astype(int)
    fx, fy = x - j, yq - i
    return ((1 - fy) * ((1 - fx) * F[i, j] + fx * F[i, j + 1])
            + fy * ((1 - fx) * F[i + 1, j] + fx * F[i + 1, j + 1]))


def _atoms_from_points(P, rho, J, n, X, Y, size_k=0.19):
    """Turn atom centres into full atom parameters, continuously.

    Size comes from the mean distance to the three nearest atoms, orientation
    and elongation from the structure tensor. Both are continuous in the atom
    positions and in the image.

    All sizes are then rescaled by ONE factor so the median matches
    0.19/sqrt(N), which is what the quadtree produces at both budgets tested
    (2.07px against a 10.67px lattice spacing at N=36, 1.01 against 5.33 at
    N=144). Section 10.14 found initial atom size to be the largest single
    effect on final quality, so without this the arms would differ in the one
    variable that matters most and the comparison would be about scale rather
    than placement. The factor is a median of continuous quantities, so it does
    not reintroduce a discrete decision."""
    N = len(P)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    k = min(3, N - 1)
    nn = np.sort(D, axis=1)[:, :k].mean(axis=1)
    Jxx = sample(J[0], P[:, 0], P[:, 1], n)
    Jxy = sample(J[1], P[:, 0], P[:, 1], n)
    Jyy = sample(J[2], P[:, 0], P[:, 1], n)
    ang, aniso, base = np.empty(N), np.empty(N), np.empty(N)
    for i in range(N):
        M = np.array([[Jxx[i], Jxy[i]], [Jxy[i], Jyy[i]]])
        ev, evec = np.linalg.eigh(M)                 # ascending
        ang[i] = np.arctan2(evec[1, 0], evec[0, 0])  # along the edge
        aniso[i] = 1.0 + 3.0 * (ev[1] - ev[0]) / max(ev[1] + ev[0], 1e-12)
        base[i] = max(size_k * nn[i], 1.0 / n)
    sigma = base / np.sqrt(aniso)                    # geometric-mean axis
    base = base * (size_k / np.sqrt(N)) / max(np.median(sigma), 1e-12)
    return np.array([e15.theta_from_axes(P[i, 0], P[i, 1], ang[i],
                                         base[i], base[i] / aniso[i])
                     for i in range(N)])


def _soft_centroids(P, rho, X, Y, tau):
    """One soft-assignment step: each atom moves to its window's centroid.

    Ordinary Lloyd assigns each pixel to its NEAREST atom, an argmin that can
    flip. This weights every pixel by distance instead, so the update is a
    weighted average and has no branch in it.

    Each pixel's weights are normalised ACROSS atoms first, so every pixel
    contributes a total of one unit of mass. Without that normalisation the
    update is a mean-shift step and the atoms migrate into density maxima and
    merge: measured at 0.02px between the closest pair at N=144, which wastes
    the budget on duplicates."""
    d2 = ((X[None, :] - P[:, 0:1]) ** 2 + (Y[None, :] - P[:, 1:2]) ** 2)
    W = np.exp(-0.5 * d2 / (tau * tau))
    W = W / np.maximum(W.sum(axis=0, keepdims=True), 1e-300)
    W = W * rho[None, :]
    m = np.maximum(W.sum(axis=1), 1e-300)
    return np.column_stack([(W @ X) / m, (W @ Y) / m])


def _init_soft(y, N, n, iters):
    X, Y = _grid(n)
    rho, J = fields(y, n)
    rho = rho.ravel()
    k = int(np.ceil(np.sqrt(N)))
    ax = (np.arange(k) + 0.5) / k
    P = np.array([[ax[j], ax[i]] for i in range(k) for j in range(k)])[:N]
    tau = 1.0 / np.sqrt(N)
    for _ in range(iters):
        P = _soft_centroids(P, rho, X, Y, tau)
    return _atoms_from_points(np.clip(P, 0.0, 1.0), rho, J, n, X, Y)


def init_softgrid(y, N, n):
    """Fixed windows, one weighted-centroid step. No decisions anywhere."""
    return _init_soft(y, N, n, iters=1)


def init_softlloyd(y, N, n):
    """Iterated soft assignment. NOT USED as an arm -- it collapses.

    Coincident atoms are a fixed point of any soft update: two atoms at the
    same place get equal responsibilities and therefore the same centroid, so
    nothing pushes them apart. Normalising responsibilities across atoms only
    halves the problem, from 0.02px between the closest pair at N=144 to 0.08.
    Kept so the check below can report the number that ruled it out."""
    return _init_soft(y, N, n, iters=10)


def init_lloyd(y, N, n, iters=10):
    """Lloyd's algorithm proper: hard Voronoi cells, density-weighted centroids.

    The argmin is back, and it is worth being precise about why that is not the
    same defect as the quadtree's. When a Lloyd boundary flips, a sliver of area
    changes cells and the two centroids move infinitesimally. When the
    quadtree's argmax flips, a DIFFERENT CELL GETS SPLIT, and every split after
    it changes. One is a continuous rule with a tie-breaking rule inside it; the
    other is a discrete rule. Hard assignment also keeps the atoms apart, which
    is exactly what the soft version fails to do."""
    X, Y = _grid(n)
    rho, J = fields(y, n)
    rho = rho.ravel()
    k = int(np.ceil(np.sqrt(N)))
    ax = (np.arange(k) + 0.5) / k
    P = np.array([[ax[j], ax[i]] for i in range(k) for j in range(k)])[:N]
    for _ in range(iters):
        d2 = ((X[None, :] - P[:, 0:1]) ** 2 + (Y[None, :] - P[:, 1:2]) ** 2)
        who = np.argmin(d2, axis=0)
        Pn = P.copy()
        for i in range(N):
            m = who == i
            w = rho[m].sum()
            if w > 1e-300:                    # an empty cell keeps its atom
                Pn[i] = [(rho[m] * X[m]).sum() / w, (rho[m] * Y[m]).sum() / w]
        P = Pn
    return _atoms_from_points(np.clip(P, 0.0, 1.0), rho, J, n, X, Y)


INITS = {"structure": e15.init_structure,
         "softgrid": init_softgrid,
         "lloyd": init_lloyd}


# ------------------------------------------------------------------- parts
def part_a(imgs, n, N, log=print):
    """Do the INITIALISATIONS move? No optimisation involved.

    Reports median AND max matched displacement. The median alone is
    misleading here: the failure mode being hunted is a few split decisions
    flipping and sending a few atoms a long way, which a median over 36 atoms
    hides completely."""
    log(f"  N={N}: displacement of the initialisation alone, before Adam, as "
        f"median / max over atoms")
    log(f"      {'init':>11} {'noise 30dB':>16} {'shift 1px':>16} "
        f"{'scale x1.01':>16}")
    rng = np.random.default_rng(11)
    rows = []
    for name, fn in INITS.items():
        acc = {}
        for (nm, y) in imgs:
            th0 = fn(y, N, n)
            for (label, y1, sh) in e16.perturbations(y, n, rng):
                t = fn(y1, N, n).copy()
                t[:, 0] -= sh / n
                acc.setdefault(label, []).append(e13.matched(th0, t, n))
        cells = []
        for k in ("noise 30dB", "shift 1px", "scale x1.01"):
            a = np.array(acc[k])
            cells.append(f"{np.median(a[:, 0]):6.2f} /{np.median(a[:, 1]):6.2f}")
        log(f"      {name:>11} " + " ".join(f"{c:>16}" for c in cells))
        rows.append(dict(init=name, **{k: [float(np.median(np.array(v)[:, 0])),
                                           float(np.median(np.array(v)[:, 1]))]
                                       for k, v in acc.items()}))
    return rows


def part_b(imgs, n, N, log=print):
    """Does the stability survive 4000 Adam steps? Each arm gets its own floor."""
    X, Y = _grid(n)
    log(f"  N={N}: displacement after 4000 Adam steps, against each arm's own "
        f"restart floor")
    log(f"      {'init':>11} {'floor':>10} {'noise 30dB':>12} "
        f"{'shift 1px':>12} {'scale x1.01':>12} {'PSNR':>7}")
    rng = np.random.default_rng(11)
    rows = []
    for name, fn in INITS.items():
        acc, ps = {}, []
        for (nm, y) in imgs:
            th_ref = e16.encode(y, N, n, X, Y, th0=fn(y, N, n))
            ps.append(e16.fit_psnr(th_ref, y, X, Y, n))
            floor = e16.encode(y, N, n, X, Y, th0=th_ref)
            acc.setdefault("floor", []).append(
                e15.matched_px(th_ref, floor, n))
            for (label, y1, sh) in e16.perturbations(y, n, rng):
                t = e16.encode(y1, N, n, X, Y, th0=fn(y1, N, n)).copy()
                t[:, 0] -= sh / n
                acc.setdefault(label, []).append(e15.matched_px(th_ref, t, n))
        log(f"      {name:>11} {np.median(acc['floor']):7.2f} px " + " ".join(
            f"{np.median(acc[k]):9.2f} px" for k in
            ("noise 30dB", "shift 1px", "scale x1.01"))
            + f" {np.mean(ps):7.3f}")
        rows.append(dict(init=name, psnr=float(np.mean(ps)),
                         **{k: float(np.median(v)) for k, v in acc.items()}))
    return rows


def part_c(imgs, n, N, log=print):
    """Quality, so a stable-but-bad encoder cannot pass as an improvement."""
    X, Y = _grid(n)
    log(f"  N={N}: PSNR after 4000 Adam steps")
    log(f"      {'init':>11} {'PSNR':>8} {'median init sigma':>19}")
    rows = []
    for name, fn in INITS.items():
        ps, sg = [], []
        for (nm, y) in imgs:
            th0 = fn(y, N, n)
            sg.append(np.median(np.exp(-0.5 * (th0[:, 2] + th0[:, 4])) * n))
            th = e16.encode(y, N, n, X, Y, th0=th0)
            ps.append(e16.fit_psnr(th, y, X, Y, n))
        log(f"      {name:>11} {np.mean(ps):8.3f} {np.median(sg):16.2f} px")
        rows.append(dict(init=name, psnr=float(np.mean(ps))))
    return rows


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    n = 64
    imgs = e14.image_set(n, np.random.default_rng(5), n_nat=2, n_cart=1, log=log)
    log("# E17 (U30): does a continuously varying placement rule fix the")
    log("# instability section 10.15 traced to the initialisation?")
    log("")

    y0 = imgs[0][1]
    for name, fn in INITS.items():
        th = fn(y0, 36, n)
        check(f"C1 {name} gives 36 finite atoms",
              th.shape == (36, 5) and np.isfinite(th).all())
        check(f"C2 {name} is deterministic",
              np.array_equal(th, fn(y0, 36, n)))
    # not a check: the iterated SOFT variant is expected to collapse, and this
    # records the number that ruled it out as an arm. Asserting a known failure
    # would put a permanent FAIL in the results and make the failure list
    # useless for spotting real breakage.
    for nm_, fn_ in (("softlloyd (dropped)", init_softlloyd),
                     ("lloyd", init_lloyd)):
        t_ = fn_(y0, 144, n)
        Dm = np.linalg.norm(t_[:, None, :2] - t_[None, :, :2], axis=-1) * n
        np.fill_diagonal(Dm, np.inf)
        log(f"    closest atom pair at N=144, {nm_:>19}: {Dm.min():5.2f} px "
            f"(lattice spacing {n / 12:.2f} px)")
    for name, fn in INITS.items():
        t_ = fn(y0, 144, n)
        Dm = np.linalg.norm(t_[:, None, :2] - t_[None, :, :2], axis=-1) * n
        np.fill_diagonal(Dm, np.inf)
        check(f"C3 {name} keeps its atoms apart", float(Dm.min()) > 0.5,
              f"(closest pair {Dm.min():.2f} px)")
    F = np.arange(n * n, dtype=float).reshape(n, n)
    got = sample(F, np.array([0.5]), np.array([0.5]), n)[0]
    want = F[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1].mean()
    check("C4 bilinear sampling hits the right value at a pixel corner",
          abs(got - want) < 1e-9, f"({got:.4f} vs {want:.4f})")

    log("")
    log("## A. Do the initialisations themselves move? (no optimisation)")
    for N in (36, 144):
        part_a(imgs, n, N, log=log)
    log("  The shift row is weak evidence for every arm: all three anchor to a")
    log("  fixed lattice, so none tracks a translation and the median reads ~1px")
    log("  by construction. The max is the informative number in that column.")
    log("")
    log("## B. Does it survive optimisation? (floor is per-arm)")
    t0 = time.time()
    part_b(imgs, n, 36, log=log)
    log(f"  [{time.time() - t0:.0f}s]")
    log("")
    log("## C. Quality, at a budget where placement matters")
    t0 = time.time()
    part_c(imgs, n, 144, log=log)
    log(f"  [{time.time() - t0:.0f}s]")

    log("")
    log(f"# checks failed: {FAIL if FAIL else 'none'}")
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
