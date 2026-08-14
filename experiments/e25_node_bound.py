"""E25: why exact branch-and-bound does not close (P0) on a splatting dictionary.

Section 7 of `convexification-and-N.md` proves (Theorem 11) that on a FINITE
dictionary the convex hull of the N-sparse box does depend on N -- the collapse
of Theorem 3 needs blobs that can collide, and dictionary atoms cannot. That is
the positive half of the document, and branch-and-bound is what exploits it:
its relaxation at every node is exactly Theorem 11's polytope P_{N,M}.

Section 9 reports that this route nevertheless fails here, and quotes three
numbers for it -- a node bound off by 5.7-8.9x, an incumbent amplitude of 24.0
against ||y|| = 21.2, and a 100% gap after two hundred thousand nodes. Those
came from an interactive probe that left no artifact in the repository. This
file is that probe, written down and committed, so the paragraph rests on the
same standard as everything else in section 9.

WHAT IS MEASURED. e8's node bound is

    LB(I) = (R - M*sigma)_+^2 / (2R),     R = ||residual on I||^2,
                                          sigma = sum of the k largest free
                                                  correlations, k = N - |I|

so it is non-zero -- informative -- exactly when the ratio M*sigma/R falls
below 1. M is the amplitude cap, and the smallest cap that still admits the
answer is the largest amplitude in the optimum the search returns; anything
smaller excludes it. Judging the bound at that cap gives the instrument the
best case it has on this problem, which is why the cap is re-tightened to the
returned optimum rather than left at the incumbent that started the search.

Four things come out per instance:

  (a) the root ratio at that cap, against M_crit = R/sigma, the cap at which
      the root bound would first say anything at all;
  (b) M_prune, the smaller cap at which the root bound alone would already
      certify the incumbent and the search would end without branching --
      the distance from M to M_prune is how far the instrument is from useful,
      as opposed to how far it is from non-zero;
  (c) the same ratio along the chain of nodes that force in the optimum's own
      atoms, in the order best-first search would branch on them -- so the
      answer is not merely "vacuous at the root", where every bound is weak;
  (d) the certified gap after 200,000 nodes, and the lower bound behind it.

PREDICTION, PRE-REGISTERED. The ratio never falls below 1 anywhere on the
chain. The mechanism is that a positive Gaussian bump on a coherent lattice
always leaves some atom well aligned with the residual, so sigma tracks ||r||
while R is ||r||^2, making the ratio grow roughly like M/||r|| as the search
descends. If it drops below 1 near the leaves the prediction is wrong and the
bound is merely late rather than useless; that is recorded either way.

WHAT THIS CAN SHOW. That on this dictionary no legal cap makes e8's node bound
informative, and why.

WHAT THIS CANNOT SHOW.
  (i)   Anything about a better node bound. This is e8's bound -- the Fenchel
        instrument specialised to a node -- not every bound a solver could use.
        A stronger relaxation at each node is untested here.
  (ii)  Anything off the dictionary, or at N above 6.
  (iii) That branch-and-bound is the wrong method. e8's own local-search
        addendum reaches the exhaustively verified optimum on 40 of 40 images
        at N=3; what fails is the CERTIFICATE, not the search.
  (iv)  One image size, one dictionary family, three targets. Same limits as
        section 9.
"""

import sys
import time

import numpy as np

import e2b_natural as e2b
import e4_exact_l0 as e4
from e8_branch_and_bound import solve_support, greedy_then_swap, bnb


def setup(name, n=48):
    """e8's panel, reproduced exactly: 48x48, parabolic dictionary, D=248."""
    rng = np.random.default_rng(0)
    y = e2b.target(name, n, rng)
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    _, G = e4.build_parabolic(n, specs, alpha=2.5)
    return y, G


def node_state(Gram, b, yy, I, N, M):
    """(R, sigma, ratio) at the node that has forced in the atoms I.

    Mirrors e8's node_bound and the sigma it feeds: the residual is the
    least-squares one on I, and sigma sums the k largest correlations over
    atoms not already in I."""
    if I:
        c, expl = solve_support(Gram, b, I, M)
        R = max(float(yy - expl), 0.0)
        rc = b - Gram[:, I] @ c
    else:
        R, rc = float(yy), b.copy()
    k = N - len(I)
    mag = np.abs(rc)
    if I:
        mag[list(I)] = -np.inf
    sigma = float(np.sort(mag)[::-1][:k].sum())
    ratio = M * sigma / R if R > 0 else np.inf
    return R, sigma, ratio, rc


def prefix_chain(Gram, b, yy, S, N, M):
    """Ratios at |I| = 0..N-1 along the optimum, in best-first branch order.

    Best-first search branches on the free atom with the largest residual
    correlation, so restricting that rule to the optimum's own support gives
    the sequence of nodes the search would traverse if it never guessed
    wrong -- the most favourable chain that exists."""
    I, rows = [], []
    for _ in range(N):
        R, sigma, ratio, rc = node_state(Gram, b, yy, I, N, M)
        rows.append((len(I), R, sigma, ratio))
        rest = [j for j in S if j not in I]
        I = I + [max(rest, key=lambda j: abs(rc[j]))]
    return rows


def run(targets=("cartoon", "ascent", "face"), Ns=(4, 6), n=48,
        max_nodes=200_000, time_limit=900.0, log=print):
    t0 = time.time()
    log("# E25: the node bound behind section 9's branch-and-bound paragraph.")
    log(f"# {n}x{n} image, e8's parabolic dictionary. M is the largest amplitude")
    log("# in the optimum the search returns -- the tightest cap that still")
    log("# admits its own answer, so the most favourable one that exists here.")
    log("# The bound is informative only where ratio = M*sigma/R falls below 1;")
    log("# M_crit is the cap where that happens and M_prune the smaller cap at")
    log("# which the root bound alone would already certify the incumbent.")

    rows, chains = [], []
    for name in targets:
        y, G = setup(name, n)
        D = len(G)
        Gram, b = G @ G.T, G @ y
        yy = float(y @ y)
        half = 0.5 * yy
        coh = float(np.abs(Gram - np.eye(D)).max())
        ny = float(np.sqrt(yy))
        log("")
        log(f"# target={name} D={D} coherence={coh:.4f} ||y||={ny:.2f}")
        log("   N        M   M/||y||     root R    sigma    ratio   M_crit"
            "  M_prune   B&B err%     gap%     nodes")
        for N in Ns:
            Sg, _ = greedy_then_swap(Gram, b, N, np.inf)
            c0, _ = solve_support(Gram, b, Sg, np.inf)
            M0 = float(np.abs(c0).max())
            Sg2, eg2 = greedy_then_swap(Gram, b, N, M0)
            S, err, gap, nodes, t, front = bnb(Gram, b, yy, N, M0, Sg2, eg2,
                                               max_nodes=max_nodes,
                                               time_limit=time_limit)
            # Re-tighten the cap to the returned optimum's own largest
            # amplitude. Anything smaller excludes the answer, so this is the
            # most favourable cap the bound can be judged at.
            cs, _ = solve_support(Gram, b, S, M0)
            M = float(np.abs(cs).max())
            R, sigma, ratio, _ = node_state(Gram, b, yy, [], N, M)
            m_crit = R / sigma
            m_prune = (R - np.sqrt(2 * R * err)) / sigma
            log(f"  {N:2d} {M:8.3f} {M/ny:9.3f} {R:10.2f} {sigma:8.2f} "
                f"{ratio:8.2f} {m_crit:8.3f} {m_prune:8.3f} "
                f"{100*err/half:10.4f} {100*gap:8.2f} {nodes:9d}")
            rows.append(dict(name=name, N=N, M=M, ny=ny, ratio=ratio,
                             m_crit=m_crit, m_prune=m_prune, err=err,
                             front=front, gap=gap, nodes=nodes, secs=t))
            chains.append((name, N, prefix_chain(Gram, b, yy, S, N, M)))

    log("")
    log("  the same ratio at the nodes that force in the optimum's own atoms,")
    log("  in the order best-first search would branch on them")
    log("   target  N   " + "".join(f"  |I|={k}" for k in range(max(Ns))))
    for name, N, ch in chains:
        cells = "".join(f"{r[3]:7.2f}" for r in ch)
        pad = "       " * (max(Ns) - len(ch))
        log(f"  {name:>7}  {N}   {cells}{pad}")
    deep = sum(1 for _, _, ch in chains
               if min(r[3] for r in ch) == ch[-1][3])
    log(f"  the ratio is lowest at the deepest node on {deep} of {len(chains)} "
        "chains, so depth")
    log("  helps it, but not by enough and not soon enough to prune anything.")

    log("")
    log("  checks")
    checks = []

    def check(label, ok, extra=""):
        checks.append(bool(ok))
        log(f"    {'ok  ' if ok else 'FAIL'} {label} {extra}")

    rr = [r["ratio"] for r in rows]
    crit = [r["m_crit"] / r["M"] for r in rows]
    check("B1 no cap that admits the answer makes the root bound informative",
          all(x > 1 for x in rr),
          f"(ratio {min(rr):.1f}-{max(rr):.1f} where under 1 is needed; the cap "
          f"would have to fall to {min(crit):.2f}-{max(crit):.2f} of the "
          "optimum's own largest amplitude, which excludes it)")

    fr = [r["front"] / r["err"] for r in rows]
    check("B2 the gap is the bound's doing, not a weak incumbent",
          all(x < 0.01 for x in fr),
          f"(the certified lower bound reaches {100*max(fr):.2f}% of the error "
          "actually attained, so no incumbent however good would close it)")

    amp = [r["M"] / r["ny"] for r in rows]
    check("B3 a single blob carries the energy of the whole image",
          max(amp) > 1.0,
          f"(largest amplitude / ||y|| ranges {min(amp):.2f}-{max(amp):.2f})")

    check(f"B4 {max_nodes} nodes leave the gap open",
          all(r["gap"] > 0.5 for r in rows),
          f"(certified gap {100*min(r['gap'] for r in rows):.0f}-"
          f"{100*max(r['gap'] for r in rows):.0f}%, node cap reached on "
          f"{sum(1 for r in rows if r['nodes'] >= max_nodes)} of {len(rows)})")

    flat = [r[3] for _, _, ch in chains for r in ch]
    check("B5 [PREDICTION] the ratio never falls below 1 along the chain",
          all(x > 1 for x in flat),
          f"(minimum over every prefix node: {min(flat):.2f})")

    log("")
    pr = [r["m_prune"] / r["M"] for r in rows]
    log("  reading: the bound e8 uses is a cap times a correlation sum against")
    log("  a residual energy. On this dictionary one atom already explains most")
    log("  of the image, so the cap that admits it is of the order of ||y||,")
    log("  and the product overwhelms the residual before the search starts.")
    log("  Making the bound useful rather than merely non-zero would need the")
    log(f"  cap at {min(pr):.2f}-{max(pr):.2f} of the optimum's largest amplitude.")
    log("  Theorem 11 is not what fails -- the polytope really does see N. What")
    log("  fails is that the only handle it gives is an amplitude cap, and on")
    log("  Gaussian blobs the amplitudes have no scale to cap at.")
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
