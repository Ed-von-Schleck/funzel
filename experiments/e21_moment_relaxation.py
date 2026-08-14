"""E18: the moment relaxation, the last convex route section 11 left untested.

Section 10 of `convexification-and-N.md` proves (Corollary 7) that no convex
lift beats the mass ball while its objective is the error of a LINEARLY
rendered image, and (Theorem 11) that lifting to second moments removes the gap
entirely at the cost of any description of the feasible set. Between those two
sits the practical question: what does a TRACTABLE outer approximation of that
lift buy? That is the Shor / doubly-nonnegative relaxation, and nothing in this
repository has tested it.

It deserves a test for a specific reason. Every relaxation measured so far
fails through an AMPLITUDE CAP with nothing to bite on -- big-M loose by
5.7-8.9x (e8), perspective loose by 64-86% wherever the ridge is small (e9),
and at most 0.15% of the quadratic available to make separable (e17). The
moment relaxation needs no cap: it carries ||Ac||^2 in a lifted matrix variable
X, so it is the one candidate not excluded by the shared cause.

THE FORMULATION. With c the coefficients and z in {0,1}^D the support
indicators, cardinality is imposed WITHOUT big-M by complementarity:

    sum_j z_j <= N,    c_j (1 - z_j) = 0.

Lift W = [1; c; z][1; c; z]^T and relax rank one to W >= 0:

    W[0,0] = 1,   diag(Z) = z,   diag(Y) = c,   sum z <= N,   0 <= z <= 1
    objective  0.5||y||^2 - c.(Gy) + 0.5 <Gram, X>

with X, Y, Z the c-c, c-z and z-z blocks. Two facts make this the right object.
A rank-one feasible W forces z_j^2 = z_j, hence z binary, hence c genuinely
N-sparse -- so the relaxation is exact at rank one and all slack comes from
higher-rank W. And the 2x2 principal minor on (c_j, z_j) reads

    [X_jj  c_j ]
    [c_j   z_j ]  >= 0    =>    X_jj * z_j >= c_j^2,

which is the perspective inequality, obtained here with no ridge and no cap.
The DNN variant adds Z >= 0 elementwise and the standard Boolean RLT cuts
Z_ij <= z_i, Z_ij >= z_i + z_j - 1, sum_j Z_ij <= N z_i.

THE METRIC. Reporting the gap against the exact l0 optimum alone is
misleading, because dropping the cardinality constraint entirely already gives
a free lower bound -- the unconstrained least-squares value E_LS. What matters
is how much of the distance from that free bound to the truth the relaxation
recovers:

    tightness = (bound - E_LS) / (E_opt - E_LS),

1.0 meaning exact, 0.0 meaning no better than forgetting cardinality existed.
An earlier probe of this file reported gaps against E_opt only and would have
credited the relaxation for whatever E_LS supplies for nothing.

WHY THE CONTROL IS BUILT THIS WAY. The obvious control -- run the same
relaxation on an incoherent random dictionary -- is confounded, and section 9
already documents the trap: relaxations go tight exactly where the dictionary
cannot approximate anything, so "incoherent => tight" may measure only "nothing
left to get wrong". A first probe of this file fell into it, with a random
dictionary that explained 1.5% of the image.

Leg A avoids it. The atom POSITIONS and the dictionary SIZE are held fixed on a
grid and only the atom WIDTH is swept, which moves pairwise coherence directly.
Every row is the same problem size with the same coverage, and E_opt is
reported alongside, so any tightness bought by ruining the dictionary is
visible rather than hidden.

Leg B runs the greedy decorrelation ladder of section 9 for comparability, and
pairs every decorrelated subdictionary with a RANDOM subset of the original of
the SAME SIZE. Decorrelation shrinks a splatting dictionary violently -- 81
atoms to 9 -- so without the size-matched partner a tightness change could be
dictionary size rather than coherence.

Leg C is the sharpest control available and holds even the SPAN fixed. An
orthonormal basis of the splat dictionary's own row space has the same
dimension, the same span, and therefore the same E_LS, but coherence zero.
Anything that changes between the two rows is attributable to coherence alone.
What it does not hold fixed is E_opt, since which vectors are sparse depends on
the basis; that is reported.

SOLVER, AND WHAT IS ACTUALLY CERTIFIED. The whole argument rests on the point
the solver returns being FEASIBLE: a feasible W has objective at least the SDP
optimum, so

    tightness reported  >=  tightness of the exact SDP,

which makes a LOW reported tightness conclusive and a high one merely possible.
Three things were needed to make that hold.

SCS at loose tolerance does not. At eps=1e-4 on D=96 it returned a value that
DECREASED when valid RLT cuts were added, which cannot happen for a relaxation,
and that is what prompted this rewrite. Clarabel and SCS at tight tolerance
agree to six digits on a brute-force-verifiable instance, so the formulation is
sound and those were solver noise. Clarabel is primary.

Clarabel alone does not either. At the coherent end the Gram is numerically
singular -- condition ~4e11, rank 35 of 36 -- and the returned point has
minimum eigenvalue near -1e-4. TIGHTENING Clarabel's tolerances makes that
worse, not better. So every returned point is repaired explicitly: clip the
negative spectrum, rescale so W[0,0] = 1, re-evaluate the objective there, and
report the repaired point's own residuals. On the worst row the repair moves
the objective by 1.0e-3 out of 4.3 and lands at residual 1.8e-6, so the
correction is real but far too small to matter.

Finally the reported value is the LARGEST across solvers and across repair,
which is the most generous reading available to the method. Both are feasible
points, so both are at least the SDP optimum, and taking the maximum keeps the
one-sided guarantee while giving the relaxation every benefit.

PRE-REGISTERED (M5).

WHAT THIS CAN SHOW.
  (a) Whether the moment relaxation is materially tighter than forgetting
      cardinality, on a coherent splatting dictionary.
  (a2) Whether it sees N at all -- V6 and V9 test monotonicity and strictness.
  (b) Whether any tightness it does have survives at coherences where the
      dictionary still approximates the image.
  (c) Whether size or coherence is responsible (leg B's paired rows).

WHAT THIS CANNOT SHOW.
  (i)   Stronger cuts exist -- rank-one convexification, higher Lasserre
        levels, problem-specific valid inequalities. A loose bound here is a
        statement about Shor and Shor+DNN+RLT, not about every moment method.
  (ii)  D <= 81 and N <= 3, set by what an interior-point SDP solver can do.
        Nothing here scales to section 10's budgets.
  (iii) tightness is an over-estimate (see above), so it bounds the method's
        performance from ABOVE only.
  (iv)  Leg A changes the dictionary's approximation power as it changes
        coherence. That is reported, not removed; no dictionary family varies
        one without the other.
  (v)   One image size, three targets. Same limits as section 9.
"""

import itertools
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cvxpy as cp

import e2b_natural as e2b
import e4_exact_l0 as e4
from e2_relaxation_gap import _grid, atoms, shape_bank


# ------------------------------------------------------------- dictionaries
def grid_dict(n, k, sig_px, aspect=1.0):
    """k*k isotropic atoms on a fixed grid; only the width varies.

    Positions and count are independent of sig_px, so sweeping the width moves
    pairwise coherence with the problem size and the image coverage held
    fixed. That is what makes leg A a control rather than a confound."""
    X, Y = _grid(n)
    (u, v, w), = shape_bank(np.log(np.array([sig_px / n])), (aspect,), (0.0,))
    px = (np.arange(k) + 0.5) / k
    PX, PY = np.meshgrid(px, px, indexing="ij")
    th = np.array([[cx, cy, u, v, w] for cx, cy in zip(PX.ravel(), PY.ravel())])
    G = atoms(th, X, Y)[0]
    return th, G / np.linalg.norm(G, axis=1, keepdims=True)


def orthonormalise(G):
    """Orthonormal basis of G's row space: same span, same size, coherence 0.

    E_LS is unchanged because the span is, so leg C varies coherence with
    everything the unconstrained problem can see held fixed."""
    Q, _ = np.linalg.qr(G.T)
    return Q.T[:len(G)]


def coherence(G):
    C = np.abs(G @ G.T)
    np.fill_diagonal(C, 0.0)
    return float(C.max())


# ------------------------------------------------------------- exact values
def exact_l0_brute(G, y, N):
    """Exhaustive N-subset search. Slow and obviously correct; used both as
    the reference value and to check e4's chunked enumerator."""
    best = np.inf
    for S in itertools.combinations(range(len(G)), N):
        S = list(S)
        a = np.linalg.lstsq(G[S].T, y, rcond=None)[0]
        r = y - a @ G[S]
        best = min(best, 0.5 * float(r @ r))
    return float(best)


def ls_value(G, y):
    c = np.linalg.lstsq(G.T, y, rcond=None)[0]
    r = y - c @ G
    return 0.5 * float(r @ r)


# --------------------------------------------------------------------- SDP
def sdp_bound(G, y, N, rlt, solver, **kw):
    """Shor (rlt=False) or Shor+DNN+RLT (rlt=True) bound, with the returned
    point's feasibility residuals so the reported number can be believed in
    the one direction that matters."""
    D = len(G)
    Gram = G @ G.T
    half = 0.5 * float(y @ y)
    Gy = G @ y
    W = cp.Variable((2 * D + 1, 2 * D + 1), PSD=True)
    c = W[0, 1:D + 1]
    z = W[0, D + 1:]
    X = W[1:D + 1, 1:D + 1]
    Y = W[1:D + 1, D + 1:]
    Z = W[D + 1:, D + 1:]
    cons = [W[0, 0] == 1, cp.diag(Z) == z, cp.diag(Y) == c,
            cp.sum(z) <= N, z >= 0, z <= 1]
    if rlt:
        one = np.ones((1, D))
        onec = np.ones((D, 1))
        zc = cp.reshape(z, (D, 1), order="C")
        zr = cp.reshape(z, (1, D), order="C")
        cons += [Z >= 0, Z <= zc @ one, Z >= zc @ one + onec @ zr - 1,
                 cp.sum(Z, axis=1) <= N * z]
    pr = cp.Problem(cp.Minimize(half - c @ Gy
                                + 0.5 * cp.sum(cp.multiply(Gram, X))), cons)
    t0 = time.time()
    try:
        pr.solve(solver=solver, **kw)
    except Exception as exc:                       # solver failure is data
        return dict(value=np.nan, raw=np.nan, repaired=np.nan,
                    status=f"error:{type(exc).__name__}", viol=np.nan,
                    viol_r=np.nan, lmin=np.nan, secs=time.time() - t0,
                    sumz=np.nan, rank_ratio=np.nan)
    if W.value is None:
        return dict(value=np.nan, raw=np.nan, repaired=np.nan,
                    status=str(pr.status), viol=np.nan, viol_r=np.nan,
                    lmin=np.nan, secs=time.time() - t0, sumz=np.nan,
                    rank_ratio=np.nan)
    Wv = (W.value + W.value.T) / 2
    cv, zv = Wv[0, 1:D + 1], Wv[0, D + 1:]
    Xv, Yv, Zv = Wv[1:D + 1, 1:D + 1], Wv[1:D + 1, D + 1:], Wv[D + 1:, D + 1:]
    viol = max(abs(Wv[0, 0] - 1.0),
               np.abs(np.diag(Zv) - zv).max(),
               np.abs(np.diag(Yv) - cv).max(),
               max(0.0, float(zv.sum()) - N),
               max(0.0, -zv.min()), max(0.0, zv.max() - 1.0))
    if rlt:
        viol = max(viol, max(0.0, -Zv.min()),
                   float((Zv - zv[:, None]).max()),
                   float((zv[:, None] + zv[None, :] - 1.0 - Zv).max()),
                   float((Zv.sum(axis=1) - N * zv).max()))
    ev, evec = np.linalg.eigh(Wv)
    raw = float(half - cv @ Gy + 0.5 * np.sum(Gram * Xv))

    # PSD repair. The claim that the reported value over-states the true SDP
    # optimum needs the returned point to be FEASIBLE, and on an ill-conditioned
    # Gram it is not: Clarabel returns eigenvalues near -1e-4 here, and
    # tightening its tolerances makes them worse, not better, because the Gram
    # is numerically singular (condition ~4e11 at the coherent end). So the
    # point is repaired explicitly -- clip the negative spectrum, rescale so
    # W[0,0] = 1 again -- and the objective is re-evaluated there. The repaired
    # point's own residuals are returned alongside, so a row whose repair does
    # not land back on the feasible set is visible rather than assumed away.
    neg = ev < 0
    Wp = Wv + (evec[:, neg] * (-ev[neg])) @ evec[:, neg].T if neg.any() else Wv
    if Wp[0, 0] > 0:
        Wp = Wp / Wp[0, 0]
    cp_, zp = Wp[0, 1:D + 1], Wp[0, D + 1:]
    Xp, Yp, Zp = Wp[1:D + 1, 1:D + 1], Wp[1:D + 1, D + 1:], Wp[D + 1:, D + 1:]
    viol_r = max(abs(Wp[0, 0] - 1.0),
                 np.abs(np.diag(Zp) - zp).max(),
                 np.abs(np.diag(Yp) - cp_).max(),
                 max(0.0, float(zp.sum()) - N),
                 max(0.0, -zp.min()), max(0.0, zp.max() - 1.0))
    if rlt:
        viol_r = max(viol_r, max(0.0, -Zp.min()),
                     float((Zp - zp[:, None]).max()),
                     float((zp[:, None] + zp[None, :] - 1.0 - Zp).max()),
                     float((Zp.sum(axis=1) - N * zp).max()))
    rep = float(half - cp_ @ Gy + 0.5 * np.sum(Gram * Xp))
    return dict(value=max(raw, rep), raw=raw, repaired=rep,
                status=str(pr.status), viol=float(viol), viol_r=float(viol_r),
                lmin=float(ev[0]), secs=time.time() - t0, sumz=float(zv.sum()),
                rank_ratio=float(ev[-2] / ev[-1]) if ev[-1] > 0 else np.nan)


def tightness(bound, E_ls, E_opt):
    den = E_opt - E_ls
    return np.nan if den <= 1e-12 else (bound - E_ls) / den


# ------------------------------------------------------------------- report
def evaluate(G, y, N, label, solver, kw, cross=None, log=print):
    E_ls = ls_value(G, y)
    E_opt = exact_l0_brute(G, y, N)
    half = 0.5 * float(y @ y)
    try:                                    # achievable, NOT a lower bound
        l1 = float(e4.exact_lasso(G, y, N, half)["err"])
    except Exception:
        l1 = np.nan
    w = np.linalg.eigvalsh(G @ G.T)
    out = dict(label=label, D=len(G), coh=coherence(G), N=N, half=half,
               E_ls=E_ls, E_opt=E_opt, l1=l1, den=(E_opt - E_ls) / half,
               rank=int((w > 1e-10 * w[-1]).sum()))
    for rlt in (False, True):
        r = sdp_bound(G, y, N, rlt, solver, **kw)
        key = "rlt" if rlt else "shor"
        out[key] = r
        out[key + "_t"] = tightness(r["value"], E_ls, E_opt)
    # One reported number per row: the largest feasible-point objective found,
    # over both solvers. Every check below reads THIS, not one solver's value,
    # so the checks and the headline cannot disagree.
    cands = [out["rlt"]["value"]]
    if cross is not None:
        out["cross"] = sdp_bound(G, y, N, True, cross[0], **cross[1])
        if np.isfinite(out["cross"]["value"]):
            cands.append(out["cross"]["value"])
    out["bound"] = max(cands)
    out["from_cross"] = len(cands) > 1 and cands[1] > cands[0]
    out["rlt_t"] = tightness(out["bound"], E_ls, E_opt)
    out["spread"] = (abs(cands[1] - cands[0]) / max(E_opt - E_ls, 1e-30)
                     if len(cands) > 1 else 0.0)
    # A rank-one W is a genuine feasible point of the ORIGINAL problem, so the
    # SDP optimum equals the l0 optimum there. That upgrades such a row from
    # the one-sided bound to an exact certificate, and it is what the
    # near-orthogonal rows return.
    out["rank_one"] = bool(out["rlt"]["rank_ratio"] < 1e-4)
    log(f"  {label:26s} D={out['D']:3d}/{out['rank']:3d} coh={out['coh']:.3f} "
        f"N={N} | E_opt={100*E_opt/half:6.2f}%  E_LS={100*E_ls/half:6.2f}%  "
        f"l1sel={100*out['l1']/half:6.2f}% | shor={out['shor_t']:6.3f}  "
        f"+rlt={out['rlt_t']:6.3f} (den={out['den']:.4f}) | "
        f"sumz={out['rlt']['sumz']:5.2f} ev2/ev1={out['rlt']['rank_ratio']:.1e} "
        f"repair={out['rlt']['repaired']-out['rlt']['raw']:+.1e} "
        f"violR={out['rlt']['viol_r']:.1e} lmin={out['rlt']['lmin']:.1e} "
        f"spread={out['spread']:.3f} {'RANK1' if out['rank_one'] else '     '} "
        f"{'scs' if out['from_cross'] else 'clb'} "
        f"{out['rlt']['status'][:14]:14s} "
        f"[{out['shor']['secs']+out['rlt']['secs']:.0f}s]")
    return out


def run(n=32, k=6, targets=("cartoon", "ascent", "face"),
        widths=(1.6, 2.2, 3.0, 4.2, 6.0, 8.5), Ns=(2, 3),
        base=(11, (2.5, 4.0, 6.5), (0.6, 1.0), 2),
        mus=(0.9, 0.8, 0.7, 0.6), seed=0, log=print):
    # Clarabel is an interior-point method and is the accurate one here; SCS
    # is the cross-check. The first draft had this the other way round, which
    # is how a first-order solver at eps=1e-4 came to report a bound that fell
    # when valid cuts were added.
    solver, kw = cp.CLARABEL, {}
    cross = (cp.SCS, dict(eps=1e-8, max_iters=200000))
    rows = []

    log("# E18: Shor / DNN moment relaxation of the cardinality-constrained")
    log("# least-squares problem, on splatting dictionaries.")
    log("# tightness = (bound - E_LS)/(E_opt - E_LS); 1 = exact, 0 = no better")
    log("# than dropping the cardinality constraint. It OVER-states the exact")
    log("# SDP's tightness, so low values are the conclusive ones.")

    log(f"\n{'='*118}")
    log("# LEG A -- coherence swept at FIXED size and FIXED positions.")
    log(f"# {k}x{k}={k*k} isotropic atoms on a grid; only the width changes.")
    for name in targets:
        y = e2b.target(name, n, np.random.default_rng(seed))
        log(f"\n  target={name}")
        for sig in widths:
            th, G = grid_dict(n, k, sig)
            for N in Ns:
                rows.append(evaluate(G, y, N, f"{name} sig={sig}px",
                                     solver, kw,
                                     cross if len(G) <= 40 else None, log))

    log(f"\n{'='*118}")
    log("# LEG B -- section 9's decorrelation ladder, each level paired with a")
    log("# RANDOM subset of the same size, so coherence and size are separated.")
    stride, sig_px, asp, rot = base
    th0, G0, _, _ = e4.build_dict(n, stride, sig_px, asp, rot)
    rng = np.random.default_rng(seed + 1)
    for name in targets[:2]:
        y = e2b.target(name, n, np.random.default_rng(seed))
        log(f"\n  target={name}  (full dictionary D={len(G0)} "
            f"coh={coherence(G0):.3f})")
        for mu in mus:
            idx = e4.decorrelate(G0, mu)
            if len(idx) < 4:
                continue
            sub = np.sort(rng.choice(len(G0), len(idx), replace=False))
            for tag, sel in (("decorrelated", idx), ("size-matched", sub)):
                rows.append(evaluate(G0[sel], y, 3, f"{name} mu={mu} {tag}",
                                     solver, kw,
                                     cross if len(sel) <= 40 else None, log))

    # tie exact_l0_brute to e4's chunked enumerator once
    _, Gchk = grid_dict(n, k, max(widths))
    ychk = e2b.target(targets[0], n, np.random.default_rng(seed))
    hchk = 0.5 * float(ychk @ ychk)
    r_e4 = e4.exact_l0(Gchk, ychk, 3, Gchk @ Gchk.T, Gchk @ ychk, hchk,
                       log=lambda *a: None)
    S_e4 = np.asarray(r_e4[0] if isinstance(r_e4, (tuple, list))
                      else r_e4["support"])
    a_e4 = np.linalg.lstsq(Gchk[S_e4].T, ychk, rcond=None)[0]
    rr = ychk - a_e4 @ Gchk[S_e4]
    e4_val = 0.5 * float(rr @ rr)
    brute = exact_l0_brute(Gchk, ychk, 3)
    e4_ok = abs(e4_val - brute) < 1e-9 * max(1.0, hchk)
    log(f"\n  e4 enumerator vs brute force at D={len(Gchk)}, N=3: "
        f"{e4_val:.9f} vs {brute:.9f} -> "
        f"{'agree' if e4_ok else 'DISAGREE'}")

    log(f"\n{'='*118}")
    log("# LEG C -- same span, same size, coherence removed by orthonormalising.")
    log("# E_LS is identical within a row pair by construction; only the basis")
    log("# differs, so any change in tightness is coherence and nothing else.")
    # The widest atoms are the most coherent, which is what leg C wants, but
    # past a point they are also numerically linearly dependent -- at D=36 the
    # widest width here has Gram rank 35. Orthonormalising a rank-deficient
    # dictionary invents a direction outside its span and the control stops
    # being a control, so take the widest width that is still full rank.
    wide = None
    for cand in sorted(widths, reverse=True):
        _, Gc = grid_dict(n, k, cand)
        ev = np.linalg.eigvalsh(Gc @ Gc.T)
        if int((ev > 1e-10 * ev[-1]).sum()) == len(Gc):
            wide = cand
            break
    if wide is None:
        wide = min(widths)
    log(f"  leg C uses sig={wide}px, the widest width whose Gram is full rank")
    for name in targets:
        y = e2b.target(name, n, np.random.default_rng(seed))
        _, Gs = grid_dict(n, k, wide)
        Go = orthonormalise(Gs)
        for N in Ns:
            for tag, GG in ((f"splat sig={wide}px", Gs), ("orthonormalised", Go)):
                rw = evaluate(GG, y, N, f"{name} legC {tag}",
                              solver, kw, cross, log)
                rw["e4_ok"] = e4_ok
                rows.append(rw)
    return rows


def report(rows, log=print):
    log(f"\n{'='*118}")
    log("# checks")
    checks = []

    def check(label, ok, extra=""):
        checks.append(bool(ok))
        log(f"  {'ok  ' if ok else 'FAIL'} {label} {extra}")

    ok_rows = [r for r in rows if np.isfinite(r.get("bound", np.nan))]
    tol = 1e-6

    # The SDP optimum is <= E_opt, but the number reported is the objective at
    # a FEASIBLE point, which is >= the SDP optimum. Where the relaxation is
    # exact the two coincide and a small excess over E_opt is expected, not a
    # breach. What must be small is the excess.
    exc = max(((r["bound"] - r["E_opt"]) / max(r["E_opt"] - r["E_ls"], 1e-30)
               for r in ok_rows), default=0.0)
    check("V1 no row's reported value materially exceeds the exact l0 optimum",
          exc < 0.02, f"(largest excess {100*exc:.3f}% of the tightness "
          f"denominator; a small excess is expected wherever the relaxation "
          f"is exact)")

    bad = [r for r in ok_rows
           if r["shor"]["value"] < r["E_ls"] - tol - 1e-9 * abs(r["E_ls"])]
    check("V2 bound >= unconstrained least squares on every row "
          "(<Gram,X> >= c'Gram c)", not bad, f"({len(bad)} violations)")

    bad = [r for r in ok_rows
           if r["rlt"]["value"] < r["shor"]["value"] - 1e-5 * max(1.0, r["half"])]  # same solver
    check("V3 adding valid RLT cuts never lowers the reported value", not bad,
          f"({len(bad)} violations; this is the check the first draft failed)")

    bad = [r for r in rows if r["E_opt"] < r["E_ls"] - tol]
    check("V0 exact l0 optimum >= unconstrained least squares (own arithmetic)",
          not bad, f"({len(bad)} violations)")

    worst = max((r["rlt"]["viol"] for r in ok_rows), default=np.nan)
    check("V4 returned points are feasible (max constraint violation small)",
          worst < 1e-6, f"(worst {worst:.2e})")
    worst_l = max((r["rlt"]["viol_r"] for r in ok_rows), default=np.nan)
    check("V5 the REPAIRED point is feasible (raw points are not, at the "
          "coherent end)", worst_l < 1e-4,
          f"(worst residual after repair {worst_l:.2e}; worst raw minimum "
          f"eigenvalue {min(r['rlt']['lmin'] for r in ok_rows):.2e})")

    worst_rep = max(((r["rlt"]["repaired"] - r["rlt"]["raw"])
                     / max(r["E_opt"] - r["E_ls"], 1e-30) for r in ok_rows),
                    default=np.nan)
    check("V5b the repair cannot change any conclusion", worst_rep < 0.02,
          f"(largest repair is {100*worst_rep:.3f}% of the tightness "
          f"denominator)")

    pairs, bad = 0, 0
    for r2 in ok_rows:
        if r2["N"] != 2:
            continue
        m = [r for r in ok_rows if r["label"] == r2["label"] and r["N"] == 3]
        if m:
            pairs += 1
            if m[0]["bound"] > r2["bound"] + 1e-5 * r2["half"]:
                bad += 1
    check("V6 bound is non-increasing in N (a larger budget is a weaker "
          "constraint)", bad == 0, f"({bad} of {pairs} pairs violate)")

    xs = [r for r in ok_rows if "cross" in r and np.isfinite(r["cross"]["value"])]
    spread = max((r["spread"] for r in ok_rows), default=0.0)
    check("V7 no row's solver spread is large enough to make it garbage",
          spread < 0.25, f"({len(xs)} rows cross-checked; largest disagreement "
          f"{100*spread:.1f}% of the denominator. Both solvers return feasible "
          f"points, so both over-state the SDP optimum and the larger is "
          f"reported -- spread threatens a HIGH tightness, never a low one)")

    legA = [r for r in ok_rows if "sig=" in r["label"] and "legC" not in r["label"]]
    if legA:
        lo = min(r["coh"] for r in legA)
        hi = max(r["coh"] for r in legA)
        t_lo = [r["rlt_t"] for r in legA if r["coh"] == lo]
        t_hi = [r["rlt_t"] for r in legA if r["coh"] == hi]
        marg = (min(t_lo) - max(t_hi)) if t_lo and t_hi else np.nan
        check("V14 leg A's contrast is far larger than the solver spread",
              marg > 4 * spread,
              f"(tightness {min(t_lo):.3f} at coherence {lo:.3f} against "
              f"{max(t_hi):.3f} at {hi:.3f}: margin {marg:.3f} versus spread "
              f"{spread:.3f})")

    strict, pairs2 = 0, 0
    for r2 in ok_rows:
        if r2["N"] != 2:
            continue
        m = [r for r in ok_rows if r["label"] == r2["label"] and r["N"] == 3]
        if m:
            pairs2 += 1
            if m[0]["bound"] < r2["bound"] - 1e-4 * r2["half"]:
                strict += 1
    check("V9 the bound actually responds to N (strict decrease somewhere)",
          strict > 0, f"({strict} of {pairs2} pairs strictly decrease)")

    cpairs = [r for r in ok_rows if "legC" in r["label"]]
    bad = 0
    for r in cpairs:
        if "orthonormalised" not in r["label"]:
            continue
        m = [q for q in cpairs if q["N"] == r["N"] and "splat" in q["label"]
             and q["label"].split(" legC")[0] == r["label"].split(" legC")[0]]
        if not m or abs(m[0]["E_ls"] - r["E_ls"]) > 1e-6 * max(1.0, r["half"]):
            bad += 1
    check("V10 leg C pairs share E_LS (the span really is held fixed)", bad == 0,
          f"({bad} pairs differ)")

    orth = [r for r in ok_rows if "orthonormalised" in r["label"]]
    worst_o = min((r["rlt_t"] for r in orth if np.isfinite(r["rlt_t"])),
                  default=np.nan)
    check("V13 relaxation is EXACT on an orthonormal dictionary, as it must be",
          len(orth) > 0 and worst_o > 1 - 1e-4,
          f"({len(orth)} rows, worst tightness {worst_o:.6f}). With Gram=I the "
          f"problem decouples and X_jj >= c_j^2/z_j makes the SDP optimum "
          f"0.5||y||^2 - 0.5*sum_j z_j <g_j,y>^2, maximised by z=1 on the top "
          f"N -- the true optimum. A miss here is the formulation or the "
          f"solver, not the dictionary.")

    ref = [r for r in rows if "legC" in r["label"] and "splat" in r["label"]]
    check("V11 brute-force optimum agrees with e4's enumerator",
          bool(ref) and ref[0].get("e4_ok", False),
          "(ties these numbers to section 9's machinery)")

    thin = [r for r in ok_rows if r["den"] < 1e-3]
    check("V12 tightness denominators are large enough to be meaningful",
          not thin, f"({len(thin)} rows with E_opt-E_LS below 0.1% of "
          f"0.5||y||^2; tightness there is noise)")

    st = [r for r in rows if not np.isfinite(r.get("bound", np.nan))]
    check("V8 every solve returned a usable point", not st,
          f"({len(st)} failures)" + (f" {[r['label'] for r in st][:3]}" if st else ""))

    log(f"\n  {sum(checks)}/{len(checks)} checks passed")

    log(f"\n{'='*118}")
    log("# leg B paired rows: does tightness follow coherence or size?")
    log("  Reported tightness is an UPPER bound on the truth, so a LOW value")
    log("  is conclusive and a high one is only a ceiling. Rows marked RANK1")
    log("  returned a rank-one W, which is a genuine feasible point of the")
    log("  original problem and therefore certifies exactness both ways.")
    log("")
    log("  label                         D   coh   E_opt%  tightness  spread  "
        "certified")
    for r in rows:
        if "mu=" in r["label"] or "legC" in r["label"]:
            log(f"  {r['label']:28s} {r['D']:3d}  {r['coh']:.3f}  "
                f"{100*r['E_opt']/r['half']:6.2f}%  {r['rlt_t']:9.3f}  "
                f"{r['spread']:6.3f}  "
                f"{'exact (rank one)' if r['rank_one'] else 'ceiling only'}")
    return checks


def main(out=None):
    stream = open(out, "w") if out else sys.stdout

    def log(*a):
        print(*a, file=stream)
        stream.flush()

    rows = run(log=log)
    report(rows, log=log)
    if out:
        stream.close()


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else None)
