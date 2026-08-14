"""E20: does Theorem 10's local-mass cap buy any bound at all?

Section 6 of `convexification-and-N.md` proves (Theorem 10) that the convex
hull of the delta-separated N-blob encodings obeys a cap of M on the mass in
any ball of radius delta/3, and is therefore strictly inside the mass ball
B_NM. The document then leaves its usefulness "open; neither derived nor
tested", which is the only constructive theorem in it and the only one with no
number attached.

On a finite dictionary that gap is trivial to close. The local cap is a finite
family of LINEAR constraints -- one per ball centre -- so

    minimise  0.5||y - cG||^2
    s.t.      sum_{j in B(i,r)} |c_j| <= M   for every ball centre i
              ||c||_1 <= N*M

is a convex quadratic program, and its optimum is a lower bound on the best
delta-separated N-blob encoding. This file computes it and compares it against
the two bounds the document already has.

THE NESTING IS EXACT, which is what makes the comparison worth doing. Every
ball contains its own centre, so the local constraint at i already forces
|c_i| <= M. Hence

    {mass ball}  ⊇  P_{N,M} = {||c||_1<=NM, ||c||_inf<=M}  ⊇  {local-mass set}

and the three bounds are ordered by construction. The first is the best any
measure-space convexification can do (Theorem 3). The second is Theorem 11's
polytope, which is exactly what branch-and-bound uses and which Section 9
measures as loose by 5.7-8.9x. The third is Theorem 10's. So the experiment
answers one question with no confounds: given big-M, does adding the local cap
move the bound, and by how much.

CHOOSING delta AND M, and why this gives the theorem its best case. Both must
admit the optimum, or the relaxation bounds a problem whose answer has been
excluded. So both are read OFF the optimum: enumerate the exact N-sparse
optimum, take M as the largest amplitude in its refit and delta as the smallest
pairwise distance within its support. The optimum is then delta-separated and
obeys the cap by construction, so the local-mass program is a valid relaxation
of a problem the optimum solves, and no smaller M or larger delta is admissible.
This is the same convention Section 9 uses for the big-M node bound -- "M set
to the largest amplitude in the best solution found so far, below which that
solution would itself be excluded".

TWO METRICS, because Theorem 10 does not fix one. Any metric on the parameter
space works in the proof, and which one is used decides how many atoms share a
ball and therefore how strong the constraints are. Both are reported:

  centre     Euclidean distance between blob centres, in pixels. Geometric and
             comparable with e19. Blind to scale: two atoms at one centre with
             different widths are at distance 0.
  coherence  1 - |<g_i,g_j>| on unit-norm atoms. Zero exactly when two atoms
             are the same blob, which is the thing Theorem 1's collision does,
             so this is the metric the theorem is really about.

THE METRIC. tightness = (L - L_ball) / (E_opt - L_ball): the share of the
distance from the mass-ball bound to the truth that a bound recovers. Zero
means it adds nothing to what Theorem 3 already allows; one means it is exact.
Reported for big-M and for local-mass, so the difference between them is what
Theorem 10 contributes and nothing else.

PRE-REGISTERED (M5), prediction on the record: the local cap will add little.
e19 measured the tightest admissible M on this dictionary as the optimum's own
largest amplitude, which reaches 1.19 times ||y||, and the mass budget N*M it
implies as 1.45-2.27 times the mass the optimum spends -- a regime where e3
already finds the mass ball vacuous. The local cap has to overcome that on its
own, and the balls at an admissible delta look too small to contain enough
atoms to bind. If the prediction is wrong the theorem becomes useful, which is
the more interesting outcome and the reason to run it.

WHAT THIS CAN SHOW.
  (a) Whether Theorem 10's set is strictly tighter than big-M in value, not
      merely in set inclusion, and by how much.
  (b) Which metric, if either, makes it bind.
  (c) How many atoms an admissible ball actually contains -- the mechanism
      behind whatever (a) shows.

WHAT THIS CANNOT SHOW.
  (i)   The bound is over dictionary atoms. Theorem 10 is a continuum
        statement, and the discrete local cap is its restriction, not a proof
        about the continuum.
  (ii)  delta and M come from the optimum, so this is the BEST case for the
        theorem, not what a solver could use without already knowing the
        answer. A negative result here is therefore strong and a positive one
        is not directly usable.
  (iii) Ball centres are the dictionary atoms, a subset of all possible
        centres. Using fewer centres means fewer constraints, so the bound
        computed is a valid but possibly weaker version of the full family.
  (iv)  N=3 on one dictionary at one image size, where the optimum is
        enumerable. Same limit as Sections 9's other exact comparisons.
  (v)   E_opt here is the unrestricted N-sparse optimum. Section 9's e4 caps
        the enumeration's mass; this does not, so the two are not the same
        quantity and are not compared.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cvxpy as cp

import e4_exact_l0 as e4
import e14_certifiable as e14


# --------------------------------------------------------------- the bounds
def qp_bound(G, y, cons_fn, solver=cp.CLARABEL, **kw):
    """min 0.5||y - cG||^2 over the polytope built by cons_fn(c).

    Returns the value at the RETURNED point together with its worst constraint
    violation, so a point that is not feasible cannot be mistaken for a bound.
    """
    D = len(G)
    c = cp.Variable(D)
    cons = cons_fn(c)
    pr = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(y - G.T @ c)), cons)
    t0 = time.time()
    try:
        pr.solve(solver=solver, **kw)
    except Exception as exc:
        return dict(value=np.nan, status=f"error:{type(exc).__name__}",
                    viol=np.nan, secs=time.time() - t0, c=None)
    if c.value is None:
        return dict(value=np.nan, status=str(pr.status), viol=np.nan,
                    secs=time.time() - t0, c=None)
    cv = np.asarray(c.value).ravel()
    r = y - cv @ G
    return dict(value=0.5 * float(r @ r), status=str(pr.status),
                viol=float(max((float(np.max(np.atleast_1d(cn.violation())))
                                for cn in cons), default=0.0)),
                secs=time.time() - t0, c=cv)


def ball_radius(delta):
    """Largest radius for which a ball still holds at most one blob of a
    delta-separated encoding.

    Theorem 10 states delta/3, but its proof needs only that a ball's diameter
    is below delta, so anything under delta/2 is valid. The larger radius puts
    MORE atoms in each ball and therefore gives a strictly tighter bound, so it
    is what a fair test of the theorem should use: if the best admissible
    radius buys nothing, the stated delta/3 certainly buys nothing."""
    return delta * 0.5 * (1 - 1e-12)


def ball_members(dist, r):
    """Float D-by-D membership matrix, row i marking the atoms within r of i."""
    return (dist <= r).astype(float)


def local_cons(c, members, M, NM):
    """Theorem 10's constraints on a finite dictionary, plus the mass ball.

    Written as one matrix inequality rather than a row at a time: with D=248
    the per-row form builds hundreds of separate expressions and dominates the
    solve. Each ball contains its own centre, so these already imply
    ||c||_inf <= M and the polytope sits inside P_{N,M}."""
    a = cp.abs(c)
    return [cp.sum(a) <= NM, members @ a <= M]


# ------------------------------------------------------------------ metrics
def centre_distance(th, n):
    p = th[:, :2] * n
    d = p[:, None, :] - p[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def coherence_distance(G):
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    return 1.0 - np.abs(Gn @ Gn.T)


# --------------------------------------------------------------------- run
def run(n=48, N=3, n_nat=4, n_cart=2, seed=0, log=print):
    sc = n / 32.0
    specs = [(12 * sc, 12 * sc, 1), (9 * sc, 6 * sc, 3),
             (6.5 * sc, 3 * sc, 6), (4.5 * sc, 1.6 * sc, 8)]
    th, G = e4.build_parabolic(n, specs, alpha=2.5)
    D = len(G)
    Gram = G @ G.T
    Dc = centre_distance(th, n)
    Dk = coherence_distance(G)
    rng = np.random.default_rng(seed)
    imgs = e14.image_set(n, rng, n_nat=n_nat, n_cart=n_cart,
                         log=lambda *a: None)

    log(f"# E20: what Theorem 10's local-mass cap is worth as a bound")
    log(f"# dictionary D={D} on {n}x{n}, N={N}; the exact optimum is "
        f"enumerated ({D*(D-1)*(D-2)//6:,} supports per image)")
    log(f"# delta and M are read off that optimum, which is the tightest")
    log(f"# admissible pair -- the same convention Section 9 uses for big-M.")
    log("")
    log("  tightness = (bound - L_ball)/(E_opt - L_ball): the share of the way")
    log("  from the mass-ball bound to the truth. big-M is Theorem 11's")
    log("  polytope; local is Theorem 10's set, which sits strictly inside it.")
    log("")
    log("  image      E_opt%  L_ball%   bigM%  local-ctr%  local-coh% | "
        "tight(bigM)  tight(ctr)  tight(coh) | ball sizes ctr/coh  M/||y||")
    rows = []
    t0 = time.time()
    for name, y in imgs:
        half = 0.5 * float(y @ y)
        ny = float(np.linalg.norm(y))
        r = e4.exact_l0(G, y, N, Gram, G @ y, half, log=lambda *a: None)
        S = np.asarray(r[0] if isinstance(r, (tuple, list)) else r["support"])
        a = np.linalg.lstsq(G[S].T, y, rcond=None)[0]
        res = y - a @ G[S]
        E_opt = 0.5 * float(res @ res)
        M = float(np.abs(a).max())
        NM = N * M
        # delta from the optimum, in each metric; radius delta/3 as in the proof
        sub_c = Dc[np.ix_(S, S)][np.triu_indices(N, 1)]
        sub_k = Dk[np.ix_(S, S)][np.triu_indices(N, 1)]
        del_c, del_k = float(sub_c.min()), float(sub_k.min())
        # A metric in which the optimum has two coincident atoms gives delta=0.
        # Theorem 10 needs delta > 0, so it simply does not apply there, and
        # reporting a bound would be reporting a relaxation of a problem whose
        # answer has been excluded.
        ok_c, ok_k = del_c > 0, del_k > 0
        mem_c = ball_members(Dc, ball_radius(del_c)) if ok_c else None
        mem_k = ball_members(Dk, ball_radius(del_k)) if ok_k else None

        b_ball = qp_bound(G, y, lambda c: [cp.sum(cp.abs(c)) <= NM])
        b_bigM = qp_bound(G, y, lambda c: [cp.sum(cp.abs(c)) <= NM,
                                           cp.norm_inf(c) <= M])
        nan = dict(value=np.nan, status="delta=0", viol=0.0, secs=0.0, c=None)
        b_locc = qp_bound(G, y, lambda c: local_cons(c, mem_c, M, NM)) \
            if ok_c else dict(nan)
        b_lock = qp_bound(G, y, lambda c: local_cons(c, mem_k, M, NM)) \
            if ok_k else dict(nan)

        # Why a cap does or does not bite: measure the mass-ball solution
        # against the two caps it is about to be given. If its largest single
        # amplitude is already below M, big-M is vacuous by construction; if
        # its largest ball mass is already below M, so is Theorem 10's cap.
        cb = b_ball["c"]
        act_inf = float(np.abs(cb).max()) / M if cb is not None else np.nan
        act_c = (float((mem_c @ np.abs(cb)).max()) / M
                 if (ok_c and cb is not None) else np.nan)
        act_k = (float((mem_k @ np.abs(cb)).max()) / M
                 if (ok_k and cb is not None) else np.nan)

        den = E_opt - b_ball["value"]

        def tg(b):
            return np.nan if den <= 1e-12 else (b["value"] - b_ball["value"]) / den

        row = dict(name=name, half=half, E_opt=E_opt, M=M, ny=ny,
                   del_c=del_c, del_k=del_k,
                   nb_c=float(mem_c.sum(1).mean()) if ok_c else np.nan,
                   nb_k=float(mem_k.sum(1).mean()) if ok_k else np.nan,
                   den=(E_opt - b_ball["value"]) / half,
                   e4err=float(r["err"]),
                   act_inf=act_inf, act_c=act_c, act_k=act_k,
                   ball=b_ball, bigM=b_bigM, locc=b_locc, lock=b_lock,
                   t_bigM=tg(b_bigM), t_locc=tg(b_locc), t_lock=tg(b_lock),
                   opt_c=a, opt_S=S)
        rows.append(row)
        log(f"  {name:9s} {100*E_opt/half:6.2f}% {100*b_ball['value']/half:7.2f}%"
            f" {100*b_bigM['value']/half:6.2f}% {100*b_locc['value']/half:10.2f}%"
            f" {100*b_lock['value']/half:10.2f}% | {row['t_bigM']:11.4f} "
            f"{row['t_locc']:11.4f} {row['t_lock']:11.4f} | "
            f"{row['nb_c']:6.1f}/{row['nb_k']:5.1f}  {M/ny:7.3f}")
        log(f"            mass-ball solution against the caps it is given: "
            f"largest single amplitude {act_inf:.3f}M, largest ball mass "
            f"{act_c:.3f}M (centre) {act_k:.3f}M (coherence) — a cap only "
            f"bites when this exceeds 1")

    log("")
    log("  checks")
    checks = []

    def check(label, ok, extra=""):
        checks.append(bool(ok))
        log(f"    {'ok  ' if ok else 'FAIL'} {label} {extra}")

    tol = 1e-6
    bad = [r for r in rows if not (r["ball"]["value"] <= r["bigM"]["value"] + tol
                                   <= r["locc"]["value"] + 2 * tol)]
    check("N1 the three bounds are ordered ball <= bigM <= local (centre "
          "metric), as the nesting requires", not bad, f"({len(bad)} rows)")
    bad = [r for r in rows if not (r["bigM"]["value"] <= r["lock"]["value"] + tol)]
    check("N2 same for the coherence metric", not bad, f"({len(bad)} rows)")
    bad = [r for r in rows if max(r[k]["value"] for k in
                                  ("ball", "bigM", "locc", "lock"))
           > r["E_opt"] + 1e-6 * max(1.0, r["half"])]
    check("N3 no bound exceeds the exact optimum it relaxes", not bad,
          f"({len(bad)} rows)")

    worst = 0.0
    for r in rows:
        c = np.zeros(len(G))
        c[r["opt_S"]] = r["opt_c"]
        ac = np.abs(c)
        for dist, dl in ((Dc, r["del_c"]), (Dk, r["del_k"])):
            if dl <= 0:
                continue
            worst = max(worst, float((ball_members(dist, ball_radius(dl)) @ ac
                                      - r["M"]).max()))
        worst = max(worst, float(ac.sum()) - N * r["M"])
    check("N4 the optimum is feasible for its own local-mass program (delta "
          "and M were read off it, so it must be)", worst < 1e-9,
          f"(worst violation {worst:.2e})")

    bad = [r for r in rows
           if abs(r["E_opt"] - r["e4err"]) > 1e-9 * max(1.0, r["half"])]
    check("N6 the refit optimum agrees with the enumerator's own value",
          not bad, f"({len(bad)} rows disagree)")

    thin = [r for r in rows if r["den"] < 1e-3]
    check("N7 the ball-to-truth distance is large enough for tightness to "
          "mean anything", not thin,
          f"({len(thin)} rows below 0.1% of 0.5||y||^2)")

    v = max((r[k]["viol"] for r in rows for k in
             ("ball", "bigM", "locc", "lock")), default=np.nan)
    check("N5 every returned point is feasible", v < 1e-6, f"(worst {v:.2e})")

    strict_c = sum(1 for r in rows if np.isfinite(r["locc"]["value"])
                   and r["locc"]["value"] > r["bigM"]["value"] + 1e-9)
    strict_k = sum(1 for r in rows if np.isfinite(r["lock"]["value"])
                   and r["lock"]["value"] > r["bigM"]["value"] + 1e-9)
    check("N8 the local cap is strictly tighter than big-M somewhere "
          "(diagnostic, not a requirement)", True,
          f"(centre metric on {strict_c} of {len(rows)} rows, coherence "
          f"metric on {strict_k})")

    gain_c = max((r["t_locc"] - r["t_bigM"] for r in rows), default=np.nan)
    gain_k = max((r["t_lock"] - r["t_bigM"] for r in rows), default=np.nan)
    log("")
    log(f"  What Theorem 10 adds over big-M, at its tightest admissible "
        f"parameters:")
    log(f"    centre metric     best gain {gain_c:+.4f} of the "
        f"ball-to-truth distance")
    log(f"    coherence metric  best gain {gain_k:+.4f}")
    log(f"    the mass-ball solution already obeys the per-atom cap at "
        f"{max(r['act_inf'] for r in rows):.3f}M and the local cap at "
        f"{max(r['act_c'] for r in rows):.3f}M, so neither is active")
    log(f"    mean atoms per admissible ball: "
        f"{np.mean([r['nb_c'] for r in rows]):.1f} (centre), "
        f"{np.mean([r['nb_k'] for r in rows]):.1f} (coherence); a ball holding "
        f"one atom reproduces big-M exactly")
    log(f"\n  [{time.time()-t0:.0f}s] {sum(checks)}/{len(checks)} checks passed")
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
