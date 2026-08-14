"""E19: are the optimal blobs separated, and is a local mass cap worth anything?

Section 6 of `convexification-and-N.md` proves (Theorem 10) that requiring the
blobs of an encoding to be at least delta apart survives the collision that
destroys every other repair: the convex hull of the separated family obeys a
cap of M on the mass in any ball of radius delta/3, so it depends on N and M
separately rather than only on their product. It then lists three things owed,
the first of which is whether a good encoding of an image HAS separated blobs.
If the optimum is a pair of near-duplicate atoms, the separated relaxation
excludes the answer and the route closes without any programme being written.

This settles that one, on the dictionary where the optimum is known exactly.

WHAT IS MEASURED, per image, at the enumerated N=3 optimum:

  max pairwise coherence   the largest |<g_i,g_j>| among the chosen blobs.
                           Near 1 means two of them are near-duplicates and no
                           useful delta admits the optimum.
  min centre distance      in pixels, the geometric version of the same thing.
  max|a| / ||y||           the largest amplitude against the image norm. This
                           decides whether the cap M that Theorem 10 needs is
                           small enough to constrain anything: M must be at
                           least the largest amplitude in the optimum, or the
                           optimum is excluded by its own relaxation.
  N*max|a| / mass          the mass budget N*M that the tightest admissible cap
                           implies, against the mass the optimum actually uses.
                           1.0 means every blob carries the same amplitude and
                           the budget is tight; large means one blob dominates
                           and the ball component of Theorem 10's set is slack.

The second pair is the point. e8 found the same dictionary's incumbent carrying
amplitude 24.0 against ||y|| = 21.2 -- a single blob holding the image's energy
-- which is why the big-M cap was useless there. Theorem 10's cap is local
rather than global, so it is not the same constraint, but it is set by the same
number, and if that number is large the ball part of the relaxation is loose
before the local part does anything.

PRE-REGISTERED (M5), and the prediction is on the record: separation looks
likely to hold, amplitude scale looks likely to sink it.

WHAT THIS CAN SHOW. Whether the enumerated optima are separated enough for
some delta to admit them, and how slack the implied mass budget is.

WHAT THIS CANNOT SHOW.
  (i)   N=3 on one dictionary at one image size. Whether optima stay separated
        at the budgets an encoder uses is not addressed.
  (ii)  It measures the optimum's geometry, not the relaxation. A slack mass
        budget makes Theorem 10's set loose; it does not prove the local cap
        adds nothing, which would need the programme actually solved.
  (iii) Coherence between chosen atoms is a proxy for the metric Theorem 10
        uses. They agree in direction -- both say "these two blobs are nearly
        the same blob" -- but no constant relates them here.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e4_exact_l0 as e4
import e14_certifiable as e14


def run(n=48, N=3, n_nat=8, n_cart=4, seed=0, log=print):
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    D = len(G)
    Gram = G @ G.T
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    Coh = np.abs(Gn @ Gn.T)
    rng = np.random.default_rng(seed)
    imgs = e14.image_set(n, rng, n_nat=n_nat, n_cart=n_cart, log=lambda *a: None)

    log(f"# E19: geometry of the exactly enumerated N={N} optimum")
    log(f"# dictionary D={D} on {n}x{n}; all C(D,{N})="
        f"{D*(D-1)*(D-2)//6:,} supports enumerated per image")
    log(f"# dictionary coherence {float(Coh[np.triu_indices(D,1)].max()):.3f}")
    log("")
    log("  image        err%   max coh   min ctr dist   max|a|/||y||   "
        "N*max|a|/mass    mass")
    rows = []
    t0 = time.time()
    for name, y in imgs:
        half = 0.5 * float(y @ y)
        ny = float(np.linalg.norm(y))
        r = e4.exact_l0(G, y, N, Gram, G @ y, half, log=lambda *a: None)
        S = np.asarray(r[0] if isinstance(r, (tuple, list)) else r["support"])
        a = np.linalg.lstsq(G[S].T, y, rcond=None)[0]
        res = y - a @ G[S]
        coh = max(float(Coh[i, j]) for k, i in enumerate(S) for j in S[k + 1:])
        ctr = th[S][:, :2] * n
        dist = min(float(np.linalg.norm(ctr[i] - ctr[j]))
                   for i in range(N) for j in range(i + 1, N))
        mass = float(np.abs(a).sum())
        mx = float(np.abs(a).max())
        rows.append(dict(name=name, err=0.5 * float(res @ res) / half, coh=coh,
                         dist=dist, amp=mx / ny, slack=N * mx / mass, mass=mass))
        log(f"  {name:9s} {100*rows[-1]['err']:6.2f}%   {coh:6.3f}   "
            f"{dist:11.2f}px   {mx/ny:12.3f}   {N*mx/mass:13.2f}   {mass:6.2f}")

    A = {k: np.array([r[k] for r in rows]) for k in
         ("coh", "dist", "amp", "slack")}
    log("")
    log(f"  median   max coh {np.median(A['coh']):.3f}   min dist "
        f"{np.median(A['dist']):.2f}px   max|a|/||y|| {np.median(A['amp']):.3f}"
        f"   N*max|a|/mass {np.median(A['slack']):.2f}")
    log(f"  range    [{A['coh'].min():.3f},{A['coh'].max():.3f}]  "
        f"[{A['dist'].min():.2f},{A['dist'].max():.2f}]px  "
        f"[{A['amp'].min():.3f},{A['amp'].max():.3f}]  "
        f"[{A['slack'].min():.2f},{A['slack'].max():.2f}]")

    log("")
    log("  checks")
    checks = []

    def check(label, ok, extra=""):
        checks.append(bool(ok))
        log(f"    {'ok  ' if ok else 'FAIL'} {label} {extra}")

    check("C1 the optima are not near-duplicate pairs",
          A["coh"].max() < 0.95,
          f"(worst pair coherence {A['coh'].max():.3f} against a dictionary "
          f"coherence of {float(Coh[np.triu_indices(D,1)].max()):.3f}) -- so "
          f"some delta admits every optimum here and Theorem 10's family is "
          f"not empty of the answer")
    check("C2 blobs are separated by a visible fraction of the image",
          A["dist"].min() > 2.0,
          f"(closest pair {A['dist'].min():.2f}px on {n}px)")
    check("C3 a single blob carries about the whole image's energy, as e8 "
          "found on this dictionary", A["amp"].max() > 0.5,
          f"(max|a|/||y|| reaches {A['amp'].max():.3f})")
    check("C4 the tightest admissible cap leaves the mass budget slack",
          A["slack"].min() > 1.2,
          f"(N*max|a|/mass is {A['slack'].min():.2f}-{A['slack'].max():.2f}; "
          f"e3 measures the mass bound as already vacuous at twice greedy's "
          f"mass)")

    log(f"\n  [{time.time()-t0:.0f}s] {sum(checks)}/{len(checks)} checks passed")
    log("")
    log("  reading: separation is not what closes Theorem 10's route -- the")
    log("  optima are comfortably separated. Amplitude scale is the problem.")
    log("  The tightest cap the relaxation may use is the optimum's own")
    log("  largest amplitude, and the mass budget N*M it implies is roughly")
    log("  twice the mass the optimum spends, which is where e3 already")
    log("  measures the ball as saying nothing. The local cap would have to")
    log("  carry the whole argument on its own, and whether it can is the")
    log("  part Theorem 10 leaves open.")
    return rows, checks


def main(out=None):
    stream = open(out, "w") if out else sys.stdout

    def log(*a):
        print(*a, file=stream)
        stream.flush()

    run(log=log)
    if out:
        stream.close()


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else None)
