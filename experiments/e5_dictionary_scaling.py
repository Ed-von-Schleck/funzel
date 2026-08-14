"""E5: does E4's dictionary result survive (a) larger N and (b) going off-grid?

E4 found dictionary design dominating everything else, but at N=3 on 32^2
images, on-grid, and -- a flaw this experiment corrects -- accounted in
DICTIONARY ATOMS rather than in splats. A difference-of-Gaussians atom is two
splats with opposite-signed coefficients, so E4's "matched size" comparison
charged the mixed dictionary half price. Everything here is budgeted in
SPLATS, which is what a 2D-GS encoder actually pays for.

Two questions, both of which could overturn E4.

(a) DOES THE ADVANTAGE SURVIVE SCALE? A structured dictionary should help most
    when atoms are scarce. With enough atoms even a badly chosen dictionary can
    cover the image, so the gap may close on its own as N grows. Budgets run
    4 -> 64 splats.

(b) DOES IT SURVIVE OFF-GRID REFINEMENT? Every dictionary here is a grid. If
    continuous refinement of the selected atoms erases the difference, then the
    dictionary is only an initialiser and the finding is much weaker: it would
    say structure buys a better starting point, not a better answer. If the gap
    persists after both sides are polished identically, the dictionary matters
    in itself.

    Note refinement DISSOLVES the DoG structure: once the two Gaussians of a DoG
    atom are free, they drift apart and become ordinary splats. That is correct
    for 2D-GS and makes (b) a clean test of the dictionary as initialiser.

PRE-REGISTERED READINGS (M5). If the ordering at N=3 (mixed < parabolic <
unstructured) holds at 32 and 64 splats and survives refinement, E4's
conclusion stands and strengthens. If refinement equalises them, the honest
claim shrinks to "structure is a better initialiser". If the advantage inverts
at large N, E4 was a small-N artifact and should be reported as such.

WHAT THIS STILL CANNOT SHOW. One instance per target, three targets, 64^2
images, greedy and exact-LASSO only -- no global optimum is computed here, so
"best" always means best among the methods run, never optimal. E4 is the only
place a true optimum was available, and only at N<=4.
"""

import sys
import time

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path
import warnings

from e2_relaxation_gap import _grid, atoms, fit_fixed_support
import e2b_natural as e2b
import e4_exact_l0 as e4

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def bounds(n):
    return (np.array([0.0, 0.0, np.log(1.5), -0.5 * n, np.log(1.5)]),
            np.array([1.0, 1.0, np.log(float(n)), 0.5 * n, np.log(float(n))]))


# ----------------------------------------------------------- dictionaries
# Every dictionary entry carries the splats that realise it, so a selected atom
# can be expanded into free Gaussians for refinement and charged its true cost.
def _pack_gauss(th, X, Y):
    G, S = [], []
    for t in th:
        a = atoms(t[None, :], X, Y)[0][0]
        nz = np.linalg.norm(a)
        if nz < 1e-12:
            continue
        G.append(a / nz)
        S.append((t[None, :].copy(), np.array([1.0 / nz])))
    return np.array(G), S


def _pack_dog(th, X, Y, k=1.6):
    G, S = [], []
    for t in th:
        a_in = atoms(t[None, :], X, Y)[0][0]
        t2 = t.copy()
        t2[2] -= np.log(k)
        t2[4] -= np.log(k)
        t2[3] = t[3] / k
        a_out = atoms(t2[None, :], X, Y)[0][0]
        n_in, n_out = np.linalg.norm(a_in), np.linalg.norm(a_out)
        if n_in < 1e-12 or n_out < 1e-12:
            continue
        d = a_in / n_in - a_out / n_out
        nz = np.linalg.norm(d)
        if nz < 1e-8:
            continue
        G.append(d / nz)
        S.append((np.vstack([t, t2]),
                  np.array([1.0 / (nz * n_in), -1.0 / (nz * n_out)])))
    return np.array(G), S


def _dedup(G, S):
    keep, Gr = [], G @ G.T
    for j in range(len(G)):
        if not keep or np.abs(Gr[j, keep]).max() < 1 - 1e-6:
            keep.append(j)
    return G[keep], [S[j] for j in keep]


def make_dicts(n):
    X, Y = _grid(n)
    sc = n / 32.0                                   # E4's specs were for n=32
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th_par, _ = e4.build_parabolic(n, specs, alpha=2.0)
    # unstructured: a dense sweep of the same scale range, no structure
    th_uns, _, _, _ = e4.build_dict(n, max(1, int(4 * sc)),
                                    (2.0 * sc, 3.5 * sc, 6.0 * sc), (0.5, 1.0), 3)
    Gu, Su = _dedup(*_pack_gauss(th_uns, X, Y))
    Gp, Sp = _dedup(*_pack_gauss(th_par, X, Y))
    Gd, Sd = _dedup(*_pack_dog(th_par, X, Y))
    Gm, Sm = np.vstack([Gp, Gd]), Sp + Sd
    return {"gauss unstructured": (Gu, Su), "gauss parabolic": (Gp, Sp),
            "DoG parabolic": (Gd, Sd), "MIXED gauss+DoG": (Gm, Sm)}


# ------------------------------------------------------------------ methods
def greedy_splats(G, S, y, budget):
    """OMP on the grid, stopping when the next atom would exceed the SPLAT
    budget. Cost is len(splats) per atom: 1 for a Gaussian, 2 for a DoG."""
    cost = np.array([len(s[1]) for s in S])
    chosen, used = [], 0
    while True:
        r = y - (np.linalg.lstsq(G[chosen].T, y, rcond=None)[0] @ G[chosen]
                 if chosen else 0.0)
        corr = np.abs(G @ r)
        corr[chosen] = -np.inf
        corr[cost > budget - used] = -np.inf
        if not np.isfinite(corr).any() or corr.max() <= 0:
            break
        j = int(np.argmax(corr))
        chosen.append(j)
        used += cost[j]
        if used >= budget:
            break
    c = np.linalg.lstsq(G[chosen].T, y, rcond=None)[0]
    r = y - c @ G[chosen]
    return np.array(chosen), c, 0.5 * float(r @ r), int(used)


def lasso_splats(G, S, y, budget):
    """Exact LARS path, taking the least-regularised point within budget."""
    cost = np.array([len(s[1]) for s in S])
    _, _, coefs = lars_path(G.T, y, method="lasso")
    best = None
    for j in range(coefs.shape[1]):
        sup = np.flatnonzero(coefs[:, j])
        if len(sup) == 0 or cost[sup].sum() > budget:
            continue
        cd = np.linalg.lstsq(G[sup].T, y, rcond=None)[0]
        r = y - cd @ G[sup]
        best = (sup, cd, 0.5 * float(r @ r), int(cost[sup].sum()))
    return best


def expand(S, idx, c):
    """Selected dictionary atoms -> free splats (theta, amplitude)."""
    th, amp = [], []
    for j, cj in zip(idx, c):
        t, w = S[j]
        th.append(t)
        amp.append(cj * w)
    return np.vstack(th), np.concatenate(amp)


def refine(th, amp, y, X, Y, n, maxiter=900):
    lo, hi = bounds(n)
    th = np.clip(th, lo, hi)
    _, _, f = fit_fixed_support(amp, th, y, X, Y, 0.0, lo, hi, maxiter=maxiter)
    return float(f)


# --------------------------------------------------------------- experiment
def run(name, n=64, budgets=(4, 8, 16, 32, 64), seed=0, log=print):
    X, Y = _grid(n)
    rng = np.random.default_rng(seed)
    y = e2b.target(name, n, rng)
    half = 0.5 * float(y @ y)
    dicts = make_dicts(n)
    pc = lambda e: 100.0 * e / half
    log(f"# target={name} n={n} 0.5||y||^2={half:.3f}")
    for k, (G, S) in dicts.items():
        log(f"#   {k:20s} D={len(G):5d}  splats/atom="
            f"{sorted({len(s[1]) for s in S})}")
    log("")
    log("  error as % of 0.5||y||^2, budgeted in SPLATS (a DoG atom costs 2)")
    log(f"{'budget':>7} {'dictionary':>20} {'greedy':>9} {'+refined':>9} "
        f"{'lasso':>9} {'used':>5}")
    rows = []
    for B in budgets:
        for k, (G, S) in dicts.items():
            t0 = time.time()
            idx, c, eg, used = greedy_splats(G, S, y, B)
            th, amp = expand(S, idx, c)
            er = refine(th, amp, y, X, Y, n)
            ml = lasso_splats(G, S, y, B)
            el = pc(ml[2]) if ml else float("nan")
            log(f"{B:7d} {k:>20} {pc(eg):9.4f} {pc(er):9.4f} {el:9.4f} "
                f"{used:5d}   [{time.time()-t0:.0f}s]")
            rows.append(dict(budget=B, dict=k, greedy=eg, refined=er,
                             lasso=ml[2] if ml else np.nan, used=used))
        best_g = min((r for r in rows if r["budget"] == B), key=lambda r: r["greedy"])
        best_r = min((r for r in rows if r["budget"] == B), key=lambda r: r["refined"])
        log(f"{'':7} {'-> best':>20} {best_g['dict']:>28} on-grid | "
            f"{best_r['dict']} refined")
    return rows


def main(targets=("cartoon", "ascent", "face"), n=64,
         budgets=(4, 8, 16, 32, 64), out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E5: does E4's dictionary result survive larger N and off-grid refinement?")
    log(f"# params: n={n} budgets(splats)={budgets} targets={targets}")
    for t in targets:
        log("")
        log("=" * 86)
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
