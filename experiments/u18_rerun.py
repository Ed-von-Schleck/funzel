"""U18: re-run E2 and E2b at DOCUMENTED parameters and commit the output.

Section 9's tables cannot be reproduced from the repository. Their parameters
were never recorded, the committed defaults do not match them, and no raw
output was kept. A code review before this run found the problem is worse than
"the parameters are missing":

  * e2's rng is shared and consumed SEQUENTIALLY inside the ratio loop. Each
    (P0) restart draws five more values, so changing n_restarts changes the
    ground-truth atoms drawn for every LATER ratio. The restart count is not a
    nuisance parameter -- it silently changes what is being fitted. Exact
    reproduction would need all of (n, K, u_px, ratios, n_restarts, seed), and
    guessing six parameters until a table matches is not evidence of anything.
  * e2 computes and prints BLpolish, but section 9.1's table omits that column.
    It is the one that says whether the gap is a support-SELECTION failure or
    merely imprecise placement that local refinement repairs, so its absence
    changes how the headline number should be read.

So this does not attempt to recover the old parameters. It runs both
experiments at parameters written down here, records them beside the numbers,
and asks the only question that matters: does the QUALITATIVE finding survive?
For section 9.1 that is the non-monotone hump in separation; for section 9.2 it
is the out-of-model penalty being positive and falling with budget.

WHAT THIS CAN SHOW. Whether section 9's conclusions rest on reproducible
observations, and what the omitted BLpolish column does to their reading.

WHAT THIS CANNOT SHOW. Agreement with the original tables, since those are not
reproducible by construction. A row that differs is not evidence the original
was wrong -- it is a different instance.
"""

import io
import sys
import time
from contextlib import redirect_stdout

import numpy as np

# Every parameter is recorded here and echoed into the results file.
# n and u_px chosen so that EVERY ratio keeps the centre lattice inside the
# image: (ceil(sqrt(K))-1)*r_max*u_px <= n, i.e. 30*4 = 120 <= 128. At e2's own
# defaults (n=96, u_px=6) the r=30 and r=15 rows fall off-image.
E2_PARAMS = dict(n=128, K=4, u_px=4.0, ratios=(30.0, 15.0, 8.0, 4.0, 2.0, 1.0),
                 n_restarts=2, seed=0)
E2B_PARAMS = dict(n=64, budgets=(8, 16, 32), n_restarts=2, seed=0)
E2B_TARGETS = ("cartoon", "ascent", "face")


def main(out="results/u18.txt"):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("# U18: E2 and E2b re-run at documented parameters.")
    log("# The original section 9 tables are NOT reproducible -- parameters were")
    log("# never recorded, and n_restarts perturbs the targets through a shared")
    log("# rng, so this is a fresh instance rather than a replication.")
    log(f"# E2  params: {E2_PARAMS}")
    log(f"# E2b params: {E2B_PARAMS} targets={E2B_TARGETS}")
    log("")

    import e2_relaxation_gap as e2
    import e2b_natural as e2b

    log("=" * 78)
    log("# E2 (section 9.1): in-model targets, separation sweep")
    log("# NOTE the BLpolish column, which section 9.1's table omitted.")
    t0 = time.time()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rows = e2.run(**E2_PARAMS)
        for ln in buf.getvalue().splitlines():
            log(ln)
        log(f"# E2 completed in {time.time()-t0:.0f}s")
        db = [r["E_db"] / (0.5 * r["ynorm"]) * 100 for r in rows]
        pol = [r["E_pol"] / (0.5 * r["ynorm"]) * 100 for r in rows]
        rs = [r["r"] for r in rows]
        peak = int(np.argmax(db))
        log("")
        log(f"# qualitative check -- section 9.1's finding is a NON-MONOTONE hump:")
        log(f"#   debiased error peaks at r={rs[peak]} ({db[peak]:.2f}%), with "
            f"{db[0]:.2f}% at r={rs[0]} and {db[-1]:.2f}% at r={rs[-1]}")
        log(f"#   hump reproduced: "
            f"{peak not in (0, len(db)-1) and db[peak] > max(db[0], db[-1]) * 2}")
        log(f"#   same rows AFTER polish: " +
            "  ".join(f"r={r:g}:{p:.2f}%" for r, p in zip(rs, pol)))
    except Exception:
        import traceback
        for ln in traceback.format_exc().splitlines():
            log("#   " + ln)

    for name in E2B_TARGETS:
        log("")
        log("=" * 78)
        log(f"# E2b (section 9.2): {name}")
        t0 = time.time()
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rows = e2b.run(name=name, **E2B_PARAMS)
            for ln in buf.getvalue().splitlines():
                log(ln)
            pen = [(r["E_db"] - r["E_ref"]) / r["half"] * 100 for r in rows]
            log(f"# penalty by budget: " +
                "  ".join(f"N={r['N']}:{p:.2f}%" for r, p in zip(rows, pen)))
            log(f"# positive and falling with budget: "
                f"{all(p > 0 for p in pen) and pen == sorted(pen, reverse=True)}")
            log(f"# completed in {time.time()-t0:.0f}s")
        except Exception:
            import traceback
            for ln in traceback.format_exc().splitlines():
                log("#   " + ln)

    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/u18.txt")
