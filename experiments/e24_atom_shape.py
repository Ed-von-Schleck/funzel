"""E24: why a splatting dictionary is coherent, and what would fix it.

Every negative measurement in `convexification-and-N.md` runs through the same
quantity. The relaxations are loose where the dictionary's atoms resemble one
another and tight where they do not (e21, and Section 9's coherence table), and
the amplitude scale that ruins the amplitude caps (e8, e20, e22, e23) is the
same phenomenon seen from the other side: atoms that resemble one another can
cancel, so the coefficients that fit an image with few of them are large and
opposed.

So the practical question the document keeps arriving at is: what makes a
splatting dictionary coherent, and is it fixable without leaving the model?
This file answers it with three measurements, and the answer is not the obvious
one.

WHAT IS MEASURED.

  1. The packing ladder. How many atoms survive a greedy prune to pairwise
     coherence <= mu, for a Gaussian dictionary and for a difference-of-
     Gaussians dictionary on the SAME lattice. An incoherent subset is a
     packing, so this counts how many mutually distinguishable atoms the family
     offers at each resolution. A dictionary that collapses under pruning is one
     whose redundancy IS its coherence.

  2. The DC control. The recurring diagnosis "a single blob carries the energy
     of the whole image" invites the explanation that positive bumps all carry
     the image's mean brightness, and that removing it would decouple them. That
     is testable in one line: project the constant out of every atom and remeasure.

  3. What the binding pairs look like. Among the pairs that are actually
     coherent: centre distance, size ratio, and orientation difference. This
     says which redundancy in the dictionary is responsible.

WHY DoG IS THE RIGHT COMPARISON AND NOT A CHANGE OF MODEL. A difference of two
concentric Gaussians is two splats with opposite-signed coefficients, which the
renderer already permits. It changes the parameterisation, not the model class.
It is also charged two splats per atom, and the companion document's M10 records
an earlier version of this comparison that looked good only because it was
budgeted in dictionary atoms rather than in splats. Nothing below is a claim
about encoding cost; it is a claim about the geometry of the dictionary.

PRE-REGISTERED (M5). Prediction: the DoG ladder holds up better, and the DC
control does nothing. The prediction about WHY was wrong and is left on the
record: it said the binding pairs would be concentric atoms of different sizes,
following the reasoning in e4's build_dog docstring. Measurement 3 below shows
they are not. Check S3 is written to test the prediction, and it fails.

WHAT THIS CAN SHOW. Which structural property of the atoms drives coherence,
and whether it is reachable inside the splatting model.

WHAT THIS CANNOT SHOW.
  (i)   Nothing about encoding quality or cost. A DoG atom costs two splats and
        this file does not budget anything.
  (ii)  One lattice, one image size. The packing numbers are properties of this
        dictionary family.
  (iii) Coherence is a proxy for what the relaxations need. e21 measures the
        relaxation directly; this file measures only the dictionary.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e4_exact_l0 as e4
import e20_separable_mass as e20


def unit(G):
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def profile(name, G, log):
    """Rank, coherence and separable mass for one dictionary."""
    G = unit(G)
    Gram = G @ G.T
    w, V = np.linalg.eigh(Gram)
    off = np.abs(Gram[np.triu_indices(len(G), 1)])
    ub, _, _ = e20.certified_bound(w, V, 0.0, log=lambda *a: None)
    log(f"  {name:22s} D={len(G):4d}  rank@1e-10={int((w>1e-10*w[-1]).sum()):4d}"
        f"  max coh={off.max():.4f}  median={np.median(off):.4f}"
        f"  pairs>0.9: {100*(off>0.9).mean():5.2f}%"
        f"  separable={100*ub/np.trace(Gram):.4f}%")
    return dict(name=name, G=G, off=off, rank=int((w > 1e-10 * w[-1]).sum()),
                sep=100 * ub / float(np.trace(Gram)))


def ladder(G, mus):
    G = unit(G)
    return [len(e4.decorrelate(G, mu)) for mu in mus]


def run(n=32, log=print):
    sc = 1.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, Gg = e4.build_parabolic(n, specs, alpha=2.5)
    Gd = e4.build_dog(n, specs, alpha=2.5, k=1.6)
    P = Gg.shape[1]
    one = np.ones(P) / np.sqrt(P)
    Gm = Gg - np.outer(Gg @ one, one)              # constant projected out
    Gm = Gm[np.linalg.norm(Gm, axis=1) > 1e-9]

    t0 = time.time()
    log("# E24: what makes a splatting dictionary coherent")
    log(f"# same lattice, {n}x{n}; a DoG atom is two splats with opposite signs,")
    log("# so it changes the parameterisation and not the model class.")
    log("")
    log("  1. profiles")
    pg = profile("Gaussian", Gg, log)
    pd = profile("DoG", Gd, log)
    pm = profile("Gaussian, DC removed", Gm, log)

    mus = (0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    lg, ld, lm = ladder(Gg, mus), ladder(Gd, mus), ladder(Gm, mus)
    log("")
    log("  2. packing ladder: atoms surviving a prune to pairwise coherence <= mu")
    log("     mu        " + "".join(f"{m:8.2f}" for m in mus))
    log("     Gaussian  " + "".join(f"{k:8d}" for k in lg))
    log("     DoG       " + "".join(f"{k:8d}" for k in ld))
    log("     DC removed" + "".join(f"{k:8d}" for k in lm))
    log("     DoG/Gauss " + "".join(f"{d/max(g,1):8.2f}" for d, g in zip(ld, lg)))

    # 3. what the binding pairs look like, on the Gaussian dictionary
    Gu = unit(Gg)
    C = np.abs(Gu @ Gu.T)
    iu = np.triu_indices(len(Gu), 1)
    ctr = th[:, :2] * n
    dist = np.linalg.norm(ctr[iu[0]] - ctr[iu[1]], axis=1)
    # Recover each atom's covariance from the Cholesky factor of its inverse,
    # so size and orientation can be compared directly.
    u, v, w = th[:, 2], th[:, 3], th[:, 4]
    Mm = np.zeros((len(th), 2, 2))
    Mm[:, 0, 0], Mm[:, 0, 1], Mm[:, 1, 1] = np.exp(u), v, np.exp(w)
    Sig = np.linalg.inv(np.einsum("kij,kil->kjl", Mm, Mm))
    ang = 0.5 * np.arctan2(2 * Sig[:, 0, 1], Sig[:, 0, 0] - Sig[:, 1, 1])
    dang = np.abs(((ang[iu[0]] - ang[iu[1]]) + np.pi / 2) % np.pi - np.pi / 2)
    dang = dang * 180 / np.pi
    area = np.sqrt(np.linalg.det(Sig))
    ratio = np.maximum(area[iu[0]] / area[iu[1]], area[iu[1]] / area[iu[0]])

    log("")
    log("  3. the pairs that actually bind, on the Gaussian dictionary")
    log("     threshold    pairs   med centre dist   med size ratio   med |angle|")
    hot = None
    for thr in (0.9, 0.8, 0.7):
        h = C[iu] > thr
        if thr == 0.9:
            hot = h
        log(f"     coh > {thr}   {int(h.sum()):6d}   {np.median(dist[h]):13.2f}px"
            f"   {np.median(ratio[h]):14.2f}   {np.median(dang[h]):9.1f} deg")
    log(f"     all pairs   {len(dist):6d}   {np.median(dist):13.2f}px"
        f"   {np.median(ratio):14.2f}   {np.median(dang):9.1f} deg")

    log("")
    log("  checks")
    checks = []

    def check(label, ok, extra=""):
        checks.append(bool(ok))
        log(f"    {'ok  ' if ok else 'FAIL'} {label} {extra}")

    k = mus.index(0.6)
    check("S1 the DoG lattice packs more atoms at equal coherence",
          ld[k] > lg[k], f"(at mu=0.6: {ld[k]} against {lg[k]}, "
          f"{ld[k]/max(lg[k],1):.1f}x)")
    check("S2 removing the constant does NOT decorrelate the dictionary",
          abs(pm["off"].max() - pg["off"].max()) < 0.01
          and abs(pm["sep"] - pg["sep"]) < 0.05,
          f"(max coherence {pg['off'].max():.4f} -> {pm['off'].max():.4f}, "
          f"separable share {pg['sep']:.4f}% -> {pm['sep']:.4f}%)")
    check("S3 [PREDICTION, FAILED] the binding pairs are concentric atoms of "
          "different sizes", np.median(ratio[hot]) > 1.05,
          f"(their median size ratio is {np.median(ratio[hot]):.2f} -- the "
          f"pairs are the SAME size. They sit {np.median(dist[hot]):.2f}px "
          f"apart against {np.median(dist):.2f}px overall and differ by "
          f"{np.median(dang[hot]):.0f} degrees of orientation against "
          f"{np.median(dang):.0f} overall, and none of them shares a shape. "
          f"The redundancy is ORIENTATION at fixed size and position, not "
          f"scale nesting)")

    log("")
    log("  reading. Two explanations are excluded and one survives.")
    log("  Not the mean: projecting the constant out of every atom leaves the")
    log("  maximum coherence and the separable share where they were.")
    log("  Not scale nesting: the pairs above 0.9 are the same size to two")
    log("  decimals, which refutes the prediction this file was written with.")
    log("  What they are is near-coincident atoms of equal size at different")
    log("  ORIENTATIONS -- an elongated bump rotated by 30 degrees about almost")
    log("  the same centre still shares most of its mass with the original,")
    log("  because a positive bump has nothing to cancel against. That is why")
    log("  a localised negative surround helps and a constant offset does not,")
    log("  and it is a property of positive bumps that no choice of lattice")
    log("  removes: the orientations have to be there for the dictionary to fit")
    log("  edges at all.")
    log(f"\n  [{time.time()-t0:.0f}s] {sum(checks)}/{len(checks)} checks passed")
    return checks


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
