"""Known-answer checks for E5, E6 and E7.

verify_primitives.py covers the shared numerical primitives. It does NOT cover
experiment-level logic, and that is where this session's two worst bugs actually
lived: E4's splat-accounting error (a fair-comparison mistake, arithmetically
perfect) and E8's -inf/abs masking (correct arithmetic, wrong selection).
Neither would have been caught by any primitive check.

E5, E6 and E7 are the three experiments with no independent reference at all --
no known answer and no redundant path. Their headline numbers are therefore
unfalsified rather than verified. This file supplies the missing references by
constructing cases whose answer is known in advance:

  * in-model targets, built as an exact combination of dictionary atoms, where
    the optimal error at the matching budget is exactly 0;
  * analytic identities the code must satisfy exactly (a Gaussian low-pass
    attenuates a pure cosine by exp(-2 pi^2 sigma^2 f^2), and preserves DC);
  * structural invariants recomputed independently (wavelet coefficient ->
    spatial position, verified against the energy centroid of the basis
    function that coefficient actually reconstructs).

THE E6 CHECK IS THE POINT OF THIS FILE. E6 concluded that wavelet placement
does not work. A transposed index in the coefficient -> (x, y) mapping, or an
inverted horizontal/vertical orientation assignment, would produce exactly that
conclusion for entirely the wrong reason, and nothing in E6 or in the primitive
suite would notice: the numbers would simply be plausibly bad. So the mapping
is checked against the actual reconstructed basis functions rather than against
my reading of the pywt documentation.

Run: python3 verify_experiments.py
"""

import sys

import numpy as np
import pywt

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def inmodel_target(G, S, rng):
    c = rng.uniform(0.7, 1.3, len(S))
    return c @ G[S], c


# ------------------------------------------------------------------------ E5
def e5_known_answer():
    """A target built from K atoms of one dictionary has optimum exactly 0 at
    K splats. The dictionary containing them must reach it; refinement must
    hold it; the others should not, which also confirms the dictionaries are
    genuinely different objects and not accidentally the same array."""
    from e2_relaxation_gap import _grid
    import e5_dictionary_scaling as e5
    n = 32
    X, Y = _grid(n)
    dicts = e5.make_dicts(n)
    rng = np.random.default_rng(0)
    G, S = dicts["gauss unstructured"]
    idx = rng.choice(len(G), 4, replace=False)
    y, _ = inmodel_target(G, idx, rng)
    half = 0.5 * float(y @ y)

    # The KNOWN-ANSWER part: on the true support the optimum is exactly 0, so
    # the expand -> refine pipeline must reproduce it. Greedy is NOT required
    # to find that support -- OMP has no such guarantee on a dictionary of
    # coherence 0.985, and demanding it would be testing a property the method
    # never claimed.
    th_t, amp_t = e5.expand(S, idx, np.zeros(len(idx)))
    Gt = G[idx]
    c_t = np.linalg.lstsq(Gt.T, y, rcond=None)[0]
    e_ls = 0.5 * float(((y - c_t @ Gt) ** 2).sum())
    th_t, amp_t = e5.expand(S, idx, c_t)
    e_ref = e5.refine(th_t, amp_t, y, X, Y, n)
    check("E5 in-model: least squares on the true support gives exactly 0",
          e_ls <= 1e-12 * half, f"{100*e_ls/half:.2e}% of 0.5||y||^2")
    check("E5 in-model: expand -> refine holds the exact solution at 0",
          e_ref <= 1e-9 * half, f"{100*e_ref/half:.2e}%")

    sel, c, eg, used = e5.greedy_splats(G, S, y, 4)
    check("E5 in-model: splat budget accounting is exact", used == 4,
          f"used {used}")
    print(f"        (reported, not asserted: greedy recovered "
          f"{len(set(sel.tolist()) & set(idx.tolist()))}/4 true atoms, "
          f"err {100*eg/half:.4f}% -- OMP is not guaranteed to recover a "
          f"support at coherence 0.985)")

    # refinement must never worsen ANY dictionary's on-grid fit
    worst = None
    for k, (Gk, Sk) in dicts.items():
        s2, c2, e2, _ = e5.greedy_splats(Gk, Sk, y, 8)
        t2, a2 = e5.expand(Sk, s2, c2)
        r2 = e5.refine(t2, a2, y, X, Y, n)
        if r2 > e2 * (1 + 1e-6) + 1e-12:
            worst = f"{k}: {100*e2/half:.4f}% -> {100*r2/half:.4f}%"
    check("E5: refinement never worsens the on-grid fit, all 4 dictionaries",
          worst is None, worst or "")


# ------------------------------------------------------------------------ E6
def e6_wavelet_mapping():
    """The coefficient -> (x, y) mapping, checked against the basis function
    the coefficient actually reconstructs.

    A single non-zero coefficient reconstructs to one wavelet basis function.
    Its energy centroid is where that coefficient LIVES, so the position
    wavelet_atoms assigns must land there. A transposed index would show up as
    the x and y errors swapping, which is why both are reported separately."""
    # haar, whose basis functions have EXACT dyadic block support. A longer
    # filter (db4) under periodization wraps around the frame, which drags the
    # energy centroid toward the image middle and makes the comparison
    # meaningless -- that confound, not the code, produced a 26px "error" in
    # the first version of this test.
    n = 32
    wav = "haar"
    L = 2
    zero = pywt.wavedec2(np.zeros((n, n)), wav, level=L, mode="periodization")
    ax = (np.arange(n) + 0.5) / n
    Yc, Xc = np.meshgrid(ax, ax, indexing="ij")     # matches _grid's convention

    worst_x = worst_y = 0.0
    rows = []
    for lev in (1, 2):
        for band in range(3):                        # 0=cH 1=cV 2=cD
            coeffs = [c.copy() if isinstance(c, np.ndarray)
                      else tuple(a.copy() for a in c) for c in zero]
            arr = coeffs[lev][band]
            m = arr.shape[0]
            i, j = m // 4, (3 * m) // 4 - 1          # an asymmetric position
            arr[i, j] = 1.0
            img = pywt.waverec2(coeffs, wav, mode="periodization")
            w = img ** 2
            cx = float((Xc * w).sum() / w.sum())
            cy = float((Yc * w).sum() / w.sum())
            # what the experiment's mapping would assign
            mx, my = (j + .5) / m, (i + .5) / m
            worst_x = max(worst_x, abs(cx - mx))
            worst_y = max(worst_y, abs(cy - my))
            # Orientation measured by which axis the basis function
            # OSCILLATES along. A function oscillating vertically has fine
            # structure in y and smooth structure in x, so the Gaussian that
            # stands in for it should be elongated along x.
            rowvar = float(np.abs(np.diff(img, axis=0)).sum())
            colvar = float(np.abs(np.diff(img, axis=1)).sum())
            rows.append((lev, "hvd"[band], cx, cy, mx, my,
                         "wider in x" if rowvar > colvar else "wider in y"))
    tol = 0.51 / n                                   # exact for haar
    check("E6 wavelet coefficient -> spatial position mapping",
          worst_x < tol and worst_y < tol,
          f"max |dx| {worst_x*n:.2f}px, max |dy| {worst_y*n:.2f}px "
          f"(tolerance {tol*n:.1f}px)")
    print("        band   centroid(x,y)      assigned(x,y)     true shape")
    for (lev, b, cx, cy, mx, my, shape) in rows:
        print(f"        L{lev} c{b.upper()}  ({cx:.3f},{cy:.3f})    "
              f"({mx:.3f},{my:.3f})   {shape}")
    # the experiment elongates cH along x and cV along y -- verify that matches
    assigned = {"h": "wider in x", "v": "wider in y"}
    mismatch = [f"c{b.upper()}" for (_, b, *_, shape) in rows
                if b in assigned and shape != assigned[b]]
    check("E6 orientation assignment (cH->x, cV->y) matches the basis functions",
          not mismatch,
          f"mismatched: {', '.join(sorted(set(mismatch)))}" if mismatch else "")


def e6_known_answer():
    """pywt round-trip, budget size, and an in-model sanity floor."""
    from e2_relaxation_gap import _grid
    import e6_wavelet_init as e6
    import e5_dictionary_scaling as e5
    n = 64
    X, Y = _grid(n)
    rng = np.random.default_rng(1)
    img = rng.normal(0, 1, (n, n))
    L = int(np.log2(n)) - 1
    for wav in ("haar", "bior2.2", "db4", "sym4"):
        c = pywt.wavedec2(img, wav, level=L, mode="periodization")
        rec = pywt.waverec2(c, wav, mode="periodization")
        if np.abs(rec - img).max() > 1e-10:
            check(f"E6 pywt round-trip ({wav})", False)
            return
    check("E6 pywt round-trip exact for all four families", True, "< 1e-10")

    sizes = []
    for B in (8, 16, 32):
        th, amp = e6.wavelet_atoms(img, n, "db4", B)
        sizes.append(th.shape[0] == B)
    check("E6 wavelet_atoms returns exactly the requested splat budget",
          all(sizes))

    dicts = e5.make_dicts(n)
    G, S = dicts["gauss unstructured"]
    idx = rng.choice(len(G), 6, replace=False)
    y, _ = inmodel_target(G, idx, rng)
    half = 0.5 * float(y @ y)
    ct = np.linalg.lstsq(G[idx].T, y, rcond=None)[0]
    th, _ = e5.expand(S, idx, ct)
    _, er = e6.fit_from_init(th, y, X, Y, n)
    check("E6 in-model: the shared refine path reaches zero on an exact target",
          er <= 1e-9 * half, f"{100*er/half:.2e}% of 0.5||y||^2")


# ------------------------------------------------------------------------ E7
def e7_known_answer():
    """The low-pass must satisfy its analytic transfer function exactly."""
    from e2_relaxation_gap import _grid
    import e7_frequency_continuation as e7
    import e5_dictionary_scaling as e5
    n = 64
    X, Y = _grid(n)
    worst = 0.0
    for f in (1, 3, 7):
        for sig in (1.0, 2.5, 6.0):
            k = np.arange(n)
            img = np.cos(2 * np.pi * f * k / n)[None, :] * np.ones((n, 1))
            out = e7.lowpass(img.ravel(), n, sig).reshape(n, n)
            want = np.exp(-2 * np.pi ** 2 * sig ** 2 * (f / n) ** 2) * img
            worst = max(worst, float(np.abs(out - want).max()))
    check("E7 lowpass matches exp(-2 pi^2 sigma^2 f^2) on pure cosines",
          worst < 1e-10, f"max abs err {worst:.2e}")

    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, n * n) + 5.0
    for sig in (0.0, 2.0, 8.0):
        ys = e7.lowpass(y, n, sig)
        if abs(ys.mean() - y.mean()) > 1e-10:
            check("E7 lowpass preserves DC", False, f"sigma={sig}")
            return
    check("E7 lowpass preserves DC exactly", True)

    dicts = e5.make_dicts(n)
    G, S = dicts["gauss unstructured"]
    idx = rng.choice(len(G), 5, replace=False)
    y, _ = inmodel_target(G, idx, rng)
    half = 0.5 * float(y @ y)
    # Start at the TRUE atoms, mildly perturbed. Starting from greedy would
    # test whether OMP recovers a support on a coherent dictionary, which it
    # does not claim to; this tests what is actually in question, namely that
    # both refinement schedules descend to the known optimum from nearby.
    ct = np.linalg.lstsq(G[idx].T, y, rcond=None)[0]
    th, amp = e5.expand(S, idx, ct)
    th = th + rng.normal(0, 0.004, th.shape)
    e_dir = e7.direct(th, amp, y, X, Y, n, 900)
    check("E7 in-model: direct refinement descends to the known optimum",
          e_dir <= 1e-6 * half, f"{100*e_dir/half:.2e}%")
    # Machinery check: a schedule with no blur must reproduce direct exactly.
    triv = e7.continuation(th, amp, y, X, Y, n, [0.0], 900, handicap=True)
    check("E7: continuation with a trivial schedule reproduces direct exactly",
          abs(triv - e_dir) <= 1e-9 * max(e_dir, 1e-30) + 1e-12,
          f"{100*triv/half:.2e}% vs {100*e_dir/half:.2e}%")
    # Reported, not asserted: the coarsest stage is destructive, and this is
    # what motivated the schedule sweep in section 10.7.
    print("        (reported: continuation from the known optimum, by schedule)")
    for sch in ([1., 0.], [2., 1., 0.], [4., 2., 1., 0.], [8., 4., 2., 1., 0.]):
        v = e7.continuation(th, amp, y, X, Y, n, sch, 900, handicap=True)
        print(f"          sigma={str(sch):22s} -> {100*v/half:.3e}%")


def main():
    print("Known-answer checks for E5, E6, E7 -- the three experiments that had")
    print("no independent reference of any kind.\n")
    for fn in (e5_known_answer, e6_wavelet_mapping, e6_known_answer,
               e7_known_answer):
        try:
            fn()
        except Exception as e:
            import traceback
            check(fn.__name__, False, f"raised {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("All known-answer checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
