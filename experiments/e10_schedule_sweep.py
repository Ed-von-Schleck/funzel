"""E10 (U22): does section 10.7's verdict on continuation depend on the schedule?

E7 tested exactly one blur schedule, sigma = 8,4,2,1,0 px, and concluded that
frequency continuation does not help. A known-answer test then showed
continuation losing a handed-in optimum by anywhere from 4e-4% to 27%,
erratically across schedules AND instances -- so a single schedule may not be
representative, and the aggregate verdict is in question.

Two failures of my own reasoning motivated this. I first attributed the loss to
the coarsest stage on the strength of one draw, and a second draw reversed which
schedule failed; that is the single-draw error rule M11 exists to prevent, made
twice in one session. So everything here is a median over seeds, and the number
of cells better/worse is reported alongside, because a median hides a bimodal
split and this quantity is visibly erratic.

DESIGN. Every method gets the handicap allocation -- the final stage, on the
true target, gets the FULL iteration budget and the coarse stages are extra --
because E7 already showed equal-compute is confounded: splitting across K stages
starves the only stage that optimises the real objective. Continuation is
therefore strictly advantaged, and a loss under that handicap cannot be a
compute artifact.

CONTROL. Schedule [0.0] is continuation with no blur at all, which must
reproduce direct refinement exactly. It is included in every run as a live
check that differences come from the schedule and not from the machinery.

WHAT THIS CAN SHOW. Whether any schedule beats direct refinement consistently,
and whether E7's verdict was an artifact of its one schedule.

WHAT THIS CANNOT SHOW. Only isotropic Gaussian low-pass schedules of this shape
are swept; a fundamentally different continuation (anisotropic, per-atom, or on
the dictionary rather than the target) is untested.
"""

import sys
import time

import numpy as np

from e2_relaxation_gap import _grid, atoms
import e2b_natural as e2b
import e5_dictionary_scaling as e5
import e7_frequency_continuation as e7

SCHEDULES = {
    "direct": None,
    "[0]": [0.0],
    "[1,0]": [1.0, 0.0],
    "[2,1,0]": [2.0, 1.0, 0.0],
    "[4,2,1,0]": [4.0, 2.0, 1.0, 0.0],
    "[8,4,2,1,0]": [8.0, 4.0, 2.0, 1.0, 0.0],
    "[16,8,4,2,1,0]": [16.0, 8.0, 4.0, 2.0, 1.0, 0.0],
}


def one_cell(y, X, Y, n, B, dicts, init, seed, total_iter=900):
    """Returns {schedule_name: error} for one (target, budget, init, seed)."""
    rng = np.random.default_rng(seed)
    if init == "greedy":
        G, S = dicts["gauss unstructured"]
        idx, c, _, _ = e5.greedy_splats(G, S, y, B)
        th, amp = e5.expand(S, idx, c)
    else:
        th = np.column_stack([
            rng.uniform(0.05, 0.95, B), rng.uniform(0.05, 0.95, B),
            np.log(rng.uniform(2.0, 20.0, B)), np.zeros(B),
            np.log(rng.uniform(2.0, 20.0, B))])
        Gr = atoms(th, X, Y)[0]
        amp = np.linalg.lstsq(Gr.T, y, rcond=None)[0]
    out = {}
    for name, sch in SCHEDULES.items():
        if sch is None:
            out[name] = e7.direct(th, amp, y, X, Y, n, total_iter)
        else:
            out[name] = e7.continuation(th, amp, y, X, Y, n, sch, total_iter,
                                        handicap=True)
    return out


def run(targets=("cartoon", "ascent", "face"), n=64, budgets=(8, 32),
        n_seeds=3, log=print):
    X, Y = _grid(n)
    dicts = e5.make_dicts(n)
    cells = []
    t0 = time.time()
    for name in targets:
        y = e2b.target(name, n, np.random.default_rng(0))
        half = 0.5 * float(y @ y)
        for B in budgets:
            for init in ("greedy", "random"):
                seeds = [0] if init == "greedy" else list(range(n_seeds))
                per_sched = {k: [] for k in SCHEDULES}
                for s in seeds:
                    r = one_cell(y, X, Y, n, B, dicts, init, 3000 + s)
                    for k, v in r.items():
                        per_sched[k].append(v / half)
                cells.append(dict(target=name, B=B, init=init,
                                  vals={k: float(np.median(v))
                                        for k, v in per_sched.items()}))
                log(f"#   {name:8s} B={B:3d} {init:7s} done "
                    f"[{time.time()-t0:.0f}s]")
    log("")
    # machinery control
    bad = [c for c in cells
           if abs(c["vals"]["[0]"] - c["vals"]["direct"])
           > 1e-9 * max(c["vals"]["direct"], 1e-30) + 1e-12]
    log(f"# control: schedule [0] reproduces direct in "
        f"{len(cells)-len(bad)}/{len(cells)} cells "
        f"{'-> machinery ok' if not bad else '-> MACHINERY MISMATCH'}")
    log("")
    log(f"{'schedule':>16} {'median vs direct':>17} {'better':>7} {'worse':>6} "
        f"{'worst cell':>11}")
    rows = []
    for name in SCHEDULES:
        if name in ("direct", "[0]"):
            continue
        rel = [(c["vals"][name] - c["vals"]["direct"])
               / max(c["vals"]["direct"], 1e-30) for c in cells]
        better = sum(1 for r in rel if r < -1e-6)
        worse = sum(1 for r in rel if r > 1e-6)
        log(f"{name:>16} {100*float(np.median(rel)):+16.2f}% {better:7d} "
            f"{worse:6d} {100*max(rel):+10.1f}%")
        rows.append(dict(schedule=name, median=float(np.median(rel)),
                         better=better, worse=worse))
    log("")
    best = min(rows, key=lambda r: r["median"])
    log(f"# best schedule by median: {best['schedule']} "
        f"({100*best['median']:+.2f}%), better in {best['better']}/"
        f"{best['better']+best['worse']} cells that moved")
    log("# negative = continuation beat direct refinement at equal final-stage")
    log("# compute, with the coarse stages given free on top.")
    return cells, rows


def main(out=None):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# E10 (U22): continuation schedule sweep, medians over seeds.")
    log(f"# schedules: {list(SCHEDULES)}")
    run(log=log)
    if out:
        open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else None)
