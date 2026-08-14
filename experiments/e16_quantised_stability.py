"""E16 (U29): can quantisation buy the stability continuous fitting does not have?

Section 10.14 left the canonical-encoder idea half alive. A deterministic
encoder costs nothing in quality -- against a scale-tuned random baseline it
scores +0.8 to +0.9 dB, inside that baseline's own seed spread -- but it is not
STABLE: 30dB of noise moves its atoms 2.7-3.3 px on a 64-pixel image, where mean
atom spacing is 10.7. Two images that differ imperceptibly get encodings that
differ visibly, so the encodings cannot be compared, which was the entire point.

Section 10.10 measured the opposite on the same perturbation: the exhaustively
enumerated GRID optimum did not move at all, 0.00 px under 30dB noise and a
one-pixel shift. The difference between the two is quantisation. A discrete
argmin cannot move a little; it either stays put or jumps a whole cell, and
under a small perturbation it stays put.

That suggests a fix that costs something known: snap the converged continuous
atoms onto a grid. Coarse enough, and two nearby images land on the same cells
and produce the SAME CODE. Too coarse, and the reconstruction suffers. The
experiment is that trade-off curve.

PART A -- the trade-off. Encode an image and its perturbed version, snap both
onto a grid of step q, refit amplitudes, and measure two things against q: what
fraction of the atoms land on identical cells, and what the snapping costs in
PSNR. Stability is measured as multiset overlap of the quantised codes, which
needs no atom-to-atom assignment and is exactly the question asked -- do the two
images produce the same code?

PART B -- where the instability lives. The encoder is deterministic, so the
movement comes from somewhere else: either the quadtree's split decisions flip
under the perturbation and the optimiser starts somewhere different, or the
local optimum itself moves. Re-encoding the perturbed image starting from the
CONVERGED solution for the clean image separates them. If it stays put, the
optimum is stable and the instability was in the initialisation and the path; if
it moves anyway, the optimum genuinely moved and no amount of careful
initialisation will help.

WHAT THIS CANNOT SHOW. 64x64 images, N=36 and 144, three images, one optimiser.
Part B's warm start is a diagnostic, not a proposal: an encoder that needs the
unperturbed image's solution to encode the perturbed one is not an encoder.
Quantising the centres is not the whole code -- scales and orientations would
also have to be quantised for a genuine discrete representation, so part A reports
all-parameter version alongside the centres-only one.
"""

import sys
import time

import numpy as np

from e2_relaxation_gap import _grid, atoms
import e14_certifiable as e14
import e15_canonical_quality as e15

FAIL = []


def check(name, ok, detail=""):
    print(f"    [{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- geometry
def axes_from_theta(th):
    """Inverse of e15.theta_from_axes: (cx, cy, angle, sL, sW) per atom.

    theta stores the upper Cholesky factor M of the inverse covariance, so the
    covariance is (M^T M)^-1 and its eigenvectors give the axes."""
    out = np.empty((len(th), 5))
    for i, t in enumerate(th):
        M = np.array([[np.exp(t[2]), t[3]], [0.0, np.exp(t[4])]])
        S = np.linalg.inv(M.T @ M)
        ev, evec = np.linalg.eigh(S)                    # ascending
        sW, sL = np.sqrt(max(ev[0], 1e-300)), np.sqrt(max(ev[1], 1e-300))
        ang = np.arctan2(evec[1, 1], evec[0, 1])        # major axis direction
        out[i] = [t[0], t[1], ang, sL, sW]
    return out


def quantise(th, n, q_px, all_params=False, ratio=1.25, n_ang=8):
    """Snap atoms onto a grid. Returns (theta, integer codes).

    Centres go onto a q_px lattice. With all_params, axis lengths go onto a
    geometric ladder of the given ratio and orientation into n_ang sectors, so
    the whole atom becomes an integer tuple -- which is what a discrete code
    has to be."""
    ax = axes_from_theta(th)
    cx = np.round(ax[:, 0] * n / q_px).astype(np.int64)
    cy = np.round(ax[:, 1] * n / q_px).astype(np.int64)
    codes = [cx, cy]
    if all_params:
        kL = np.round(np.log(np.maximum(ax[:, 3], 1e-12)) / np.log(ratio))
        kW = np.round(np.log(np.maximum(ax[:, 4], 1e-12)) / np.log(ratio))
        ka = np.round((ax[:, 2] % np.pi) / (np.pi / n_ang)) % n_ang
        codes += [kL.astype(np.int64), kW.astype(np.int64), ka.astype(np.int64)]
        sL, sW = ratio ** kL, ratio ** kW
        ang = ka * (np.pi / n_ang)
    else:
        sL, sW, ang = ax[:, 3], ax[:, 4], ax[:, 2]
    out = np.array([e15.theta_from_axes(cx[i] * q_px / n, cy[i] * q_px / n,
                                        ang[i], max(sL[i], 1e-6),
                                        max(sW[i], 1e-6))
                    for i in range(len(th))])
    return out, list(zip(*codes))


def code_overlap(a, b):
    """Fraction of one code multiset found in the other. Assignment-free."""
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values()) / max(len(a), 1)


def fit_psnr(th, y, X, Y, n):
    """Least-squares amplitudes on the given atoms, then PSNR."""
    G = atoms(th, X, Y)[0]
    c = np.linalg.lstsq(G.T, y, rcond=None)[0]
    r = y - c @ G
    return e15.psnr(0.5 * float(r @ r), n * n)


# ------------------------------------------------------------------ driver
def encode(y, N, n, X, Y, th0=None, steps=(4000,)):
    """The deterministic encoder of section 10.14: quadtree init, then Adam."""
    if th0 is None:
        th0 = e15.init_structure(y, N, n)
    return e15.adam(th0, y, X, Y, n, checkpoints=steps)[steps[-1]][1]


def perturbations(y, n, rng):
    sig = np.sqrt((y @ y) / len(y) * 1e-3)                  # 30 dB
    return [("noise 30dB", y + rng.normal(0, sig, len(y)), 0.0),
            ("shift 1px", np.roll(y.reshape(n, n), 1, axis=1).ravel(), 1.0),
            ("scale x1.01", y * 1.01, 0.0)]


def part_a(imgs, n, N, qs, log=print):
    X, Y = _grid(n)
    log(f"  N={N}: snap converged atoms onto a grid of step q, refit amplitudes")
    log(f"      {'q (px)':>7} {'PSNR':>7} {'dPSNR':>7} "
        f"{'code match: centres only':>26} {'all params':>12}")
    log(f"      {'':>7} {'':>7} {'vs free':>7} "
        f"{'noise / shift / scale':>26} {'noise':>12}")
    rng = np.random.default_rng(11)
    ref, pert = {}, {}
    t0 = time.time()
    for ii, (nm, y) in enumerate(imgs):
        ref[ii] = (y, encode(y, N, n, X, Y))
        for (label, y1, _) in perturbations(y, n, rng):
            pert[(ii, label)] = (y1, encode(y1, N, n, X, Y))
    log(f"      [{time.time() - t0:.0f}s of fitting]")
    free = float(np.mean([fit_psnr(ref[i][1], ref[i][0], X, Y, n)
                          for i in ref]))
    rows = []
    for q in qs:
        ps, ov, ov_all = [], {}, []
        for ii, (y, th) in ref.items():
            thq, code = quantise(th, n, q)
            ps.append(fit_psnr(thq, y, X, Y, n))
            for (label, _, sh) in perturbations(y, n, np.random.default_rng(0)):
                y1, th1 = pert[(ii, label)]
                t1 = th1.copy()
                t1[:, 0] -= sh / n                    # undo the known shift
                _, c1 = quantise(t1, n, q)
                ov.setdefault(label, []).append(code_overlap(code, c1))
                if label == "noise 30dB":
                    _, ca = quantise(th, n, q, all_params=True)
                    _, cb = quantise(t1, n, q, all_params=True)
                    ov_all.append(code_overlap(ca, cb))
        m = float(np.mean(ps))
        cells = " / ".join(f"{np.mean(ov[k]):.2f}"
                           for k in ("noise 30dB", "shift 1px", "scale x1.01"))
        log(f"      {q:7.2f} {m:7.3f} {m - free:+7.3f} {cells:>26} "
            f"{np.mean(ov_all):12.2f}")
        rows.append(dict(q=q, psnr=m, dpsnr=m - free,
                         overlap={k: float(np.mean(v)) for k, v in ov.items()},
                         overlap_all=float(np.mean(ov_all))))
    return rows


def part_b(imgs, n, N, log=print):
    """Warm start: is the instability in the initialisation or in the optimum?"""
    X, Y = _grid(n)
    rng = np.random.default_rng(11)
    log(f"  N={N}: re-encode the perturbed image from the CLEAN image's "
        f"converged solution")
    log(f"      {'perturbation':>14} {'cold start':>12} {'warm start':>12} "
        f"{'PSNR cold':>10} {'PSNR warm':>10}")
    acc = {}
    for (nm, y) in imgs:
        th_ref = encode(y, N, n, X, Y)          # once per image, not once per row
        for (label, y1, sh) in perturbations(y, n, rng):
            th_cold = encode(y1, N, n, X, Y)
            th_warm = encode(y1, N, n, X, Y, th0=th_ref)
            row = []
            for th in (th_cold, th_warm):
                t = th.copy()
                t[:, 0] -= sh / n               # undo the known shift
                row += [e15.matched_px(th_ref, t, n), fit_psnr(th, y1, X, Y, n)]
            acc.setdefault(label, []).append(row)
    out = []
    for label, rows in acc.items():
        m = np.median(np.array(rows), axis=0)
        log(f"      {label:>14} {m[0]:9.2f} px {m[2]:9.2f} px "
            f"{m[1]:10.3f} {m[3]:10.3f}")
        out.append(dict(perturbation=label, cold=float(m[0]), warm=float(m[2]),
                        psnr_cold=float(m[1]), psnr_warm=float(m[3])))
    return out


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    n = 64
    imgs = e14.image_set(n, np.random.default_rng(5), n_nat=2, n_cart=1, log=log)
    log("# E16 (U29): does quantisation buy stability, and what does it cost?")
    log("# Section 10.14: continuous fitting moves 2.7-3.3px under 30dB noise.")
    log("# Section 10.10: the enumerated grid optimum moved 0.00px. The")
    log("# difference is quantisation, so this prices it.")
    log("")

    # the geometry round-trip has to be exact or every code below is noise
    th = e15.init_structure(imgs[0][1], 24, n)
    ax = axes_from_theta(th)
    back = np.array([e15.theta_from_axes(*a) for a in ax])
    G1 = atoms(th, _grid(n)[0], _grid(n)[1])[0]
    G2 = atoms(back, _grid(n)[0], _grid(n)[1])[0]
    check("C1 axes_from_theta inverts theta_from_axes",
          float(np.abs(G1 - G2).max()) < 1e-8,
          f"(max atom-image difference {np.abs(G1 - G2).max():.2e})")
    thq, code = quantise(th, n, 1.0)
    check("C2 quantising at 1px moves centres by at most half a pixel",
          float(np.abs(thq[:, :2] - th[:, :2]).max() * n) <= 0.5 + 1e-9,
          f"(max {np.abs(thq[:, :2] - th[:, :2]).max() * n:.3f} px)")
    check("C3 identical inputs give identical codes",
          code_overlap(code, quantise(th, n, 1.0)[1]) == 1.0)
    check("C4 a coarse grid collapses distinct atoms into shared cells",
          len(set(quantise(th, n, 32.0)[1])) < len(set(code)),
          f"({len(set(quantise(th, n, 32.0)[1]))} cells at q=32 vs "
          f"{len(set(code))} at q=1)")

    log("")
    log("## A. The trade-off: stability bought against PSNR paid")
    for N in (36, 144):
        part_a(imgs, n, N, (0.25, 0.5, 1.0, 2.0, 4.0, 8.0), log=log)
        log("")
    log("## B. Is the instability in the initialisation or in the optimum?")
    part_b(imgs, n, 36, log=log)

    log("")
    log(f"# checks failed: {FAIL if FAIL else 'none'}")
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
