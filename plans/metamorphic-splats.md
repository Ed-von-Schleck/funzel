# Metamorphic splats: can richer primitives beat more Gaussians?

An experiment design. Self-contained and independent of the documents in `docs/` — it shares no
premises with them and none of their conclusions are assumed here.

**Status: planned, not run.** Nothing below is a result.

---

## 1. The question

2D Gaussian splatting represents an image as $\sum_i c_i\,G(\mu_i,\Sigma_i)$ — many cheap
primitives, 6 parameters each, fit by random init + Adam. The question: at a **fixed total
parameter budget** $P$, is it better to spend $P$ on more atoms ($N\uparrow$, $k=6$) or on
richer atoms ($k\uparrow$, $N\downarrow$)?

Framing that makes the question non-trivial: the curve from (many atoms, $k{=}1$) — pixels — to
(one atom, $k{=}10^6$) — an implicit neural representation — passes through Gaussian splatting
near the cheap end. Whether the distortion-at-budget curve dips in the middle is
content-dependent: an edge wants many cheap anisotropic atoms, a texture patch wants one rich
oscillatory one, a sky wants one enormous smooth one. So the design tests not a single richer
primitive but an **allocation policy** deciding, per location, where on the $N$-vs-$k$ curve to
sit.

**The core comparison is the triangle A1 / A2a / A3** (§6): incumbent Gaussians, uniformly rich
atoms, adaptively rich atoms — all at matched $P$. Every other arm exists to attribute a win or
a loss, not to answer the question. Fixes and controls below must not displace that triangle.

## 2. Why "random init + Adam" works, and what it filters out

The recipe's success with Gaussians is not a statement about the loss landscape of one atom. It
rests on population dynamics over four properties:

1. **Capture radius ≈ footprint.** A Gaussian feels residual over a region as wide as itself;
   its position gradient has no sign flips. Random init only needs *some* atom within capture
   radius of each feature.
2. **Amplitude passes through zero.** Wrong atoms fade out continuously or get pruned; errors
   are recoverable, so init quality does not matter.
3. **A built-in homotopy.** The scale parameter blunts the atom — grown large it sees a
   smoothed loss, shrunk it refines. Per-atom graduated non-convexity.
4. **Near-decoupling.** Atoms interact only where they overlap; the Hessian is nearly
   block-diagonal; Adam's per-parameter normalization handles the heterogeneous units.

The sieve for any enrichment is therefore: *when a fresh atom is spawned on a residual hotspot,
do all its parameters get informative gradients — and can it die continuously if wrong?* Three
design principles fall out:

- **Enrich linearly where possible.** Parameters that enter linearly (coefficients over a
  per-atom basis) are nearly free: given pose, they are one least-squares solve. Nonlinear
  parameters (frequency, sharpness, warp) are what wreck landscapes.
- **Every atom is born a Gaussian**, and every knob has a blunt value at which the atom *is* a
  plain Gaussian — with **no dormant parameters attached**. A parameter exists only from the
  moment it can receive gradient (§4).
- **Domain knowledge enters at birth, not during descent.** Nonlinear parameters that descent
  cannot find (a carrier frequency: capture radius in space $\sigma$ times capture radius in
  frequency $1/\sigma$ is pinned at $\sim1$ by the uncertainty principle, so position and
  frequency cannot both have wide basins) are set by a matched-filter measurement of the
  residual at unlock time. Adam polishes; birth places.

## 3. Prior art, briefly

The kernel-swap literature is active but fragmented; none of it tests the allocation question.
*Citation depth caveat: the works below are known from abstracts, project pages, and search
summaries, not full readings; before running, the two closest (3DGabSplat, Gaussian–Hermite)
should be read in full to steal their failure notes.*

- **Real Gabor splats.** Gabor Splatting for gigapixel images (SIGGRAPH 2024 poster,
  [doi:10.1145/3641234.3671081](https://dl.acm.org/doi/10.1145/3641234.3671081)); 3D Gabor
  Splatting ([arXiv:2504.11003](https://arxiv.org/abs/2504.11003), EG 2025); 3DGabSplat
  ([arXiv:2508.05343](https://arxiv.org/abs/2508.05343), ACM MM 2025, +1.35 dB over 3DGS with
  fewer primitives); Neural Gabor Splatting ([arXiv:2604.15941](https://arxiv.org/abs/2604.15941))
  retreats from explicit waves to a per-atom MLP; AdaGaR
  ([arXiv:2601.00796](https://arxiv.org/abs/2601.00796)) names the "energy instability of fixed
  Gabor functions" and compensates for it. All real-valued: phase is a descended parameter, and
  the listed workarounds are the cost of that choice showing up in practice.
- **Complex-valued splats exist only where physics forces them**: MRI (Gabor primitives with
  complex-exponential modulation, [arXiv:2603.05681](https://arxiv.org/abs/2603.05681)) and
  holography (Gaussian Wave Splatting, [arXiv:2505.06582](https://arxiv.org/abs/2505.06582);
  complex-valued 2D Gaussians, [arXiv:2511.15022](https://arxiv.org/abs/2511.15022)). Nobody
  uses quadrature amplitude as an *optimization device* for ordinary photographs.
- **Other kernel swaps**: GES's learnable exponent
  ([arXiv:2402.10128](https://arxiv.org/abs/2402.10128)), Gaussian–Hermite splatting
  ([arXiv:2408.16982](https://arxiv.org/abs/2408.16982)) — which needed a coarse-to-fine
  schedule on Hermite order, consistent with §2's basin-narrowing prediction — Student-t, Beta,
  and convex kernels. Each swaps one kernel uniformly; none allocates richness adaptively.

## 4. The unified atom

Written in real form so that parameter accounting is honest — a parameter appears below exactly
when it exists and can receive gradient:

$$f(x) \;=\;
\Big[\underbrace{\textstyle\sum_{|\alpha|\le m} c_\alpha H_\alpha(u)}_{\text{always present}}\Big]
\cos(\omega^\top u)\;+\;
\Big[\underbrace{\textstyle\sum_{|\alpha|\le m} d_\alpha H_\alpha(u)}_{\text{exists only if }\omega\text{ unlocked}}\Big]
\sin(\omega^\top u),
\quad\text{all}\times\; e^{-\frac12\|u\|^{\beta}},\quad u=A(x-\mu).$$

$A$ is the usual anisotropic pose (2 log-scales + rotation), $H_\alpha$ the Hermite basis.
While $\omega$ is locked at $0$, the $\sin$ block and all $d_\alpha$ **do not exist** — they
are not stored at zero, they are absent, so the blunt atom is exactly a 6-parameter Gaussian
with no dormant freight. Unlocking $\omega$ creates the carrier *and* one quadrature partner
$d_\alpha$ per existing $c_\alpha$; from then on the appearance pair $(c_\alpha,d_\alpha)$ is
the complex coefficient of the earlier discussions, and phase is never a descended parameter.
The order-0, $\omega$-unlocked atom is precisely the complex Gabor.

| knob | blunt value | unlock cost | serves | sieve status |
|---|---|---|---|---|
| order $m{:}\,0\to1$ | $m=0$ | +2 if $\omega$ locked; +4 if unlocked | edges, ridges, junctions | linear ⇒ marginal gain exact by projection (rest frozen) |
| carrier $\omega$ | absent ($\equiv0$) | +2 + one $d_\alpha$ per existing $c_\alpha$ (= +3 at $m{=}0$) | periodic texture | nonlinear ⇒ set by spectrogram measurement at unlock |
| exponent $\beta$ | $\beta=2$ | +1, range $[1,8]$ | region boundaries, flat fills | nonlinear but demonstrated trainable (GES) |

Hermite order and carrier are complementary, not redundant in *role*: order $m$ buys $m$
structured wiggles for $O(m)$ parameters (edges); $\omega$ buys unlimited periodic cycles for
2–3 (texture). They are, however, partially redundant in *value*: the order-1 subspace overlaps
$\partial_\omega$ of the carrier (both look like $u\cdot(\text{carrier})\cdot G$), and for an
isotropic envelope the pose rotation is redundant with the direction of $\omega$. Consequence
for the allocator: unlock gains for these knobs double-count, which §5's calibration exists to
absorb; consequence for the optimizer: intra-atom curvature coupling, which is the case for
per-atom block preconditioning if diagonal Adam stalls (recorded as a fallback, not a plan).

**Re-locking.** Every unlock is reversible: when a knob's contribution decays below threshold
($\|(c,d)\|$ beyond order 0, $|\omega|$, $|\beta-2|$), the knob re-locks and its parameters
return to the pool. Continuous death applies to knobs, not only atoms — without it, early bad
unlocks strand budget permanently.

**Anti-aliasing, analytic.** The family is closed under convolution with a Gaussian pixel
filter: polynomial × Gaussian stays polynomial × Gaussian, and a carrier passes through with
amplitude attenuation $e^{-\sigma_p^2\|\omega\|^2/2}$ plus an envelope correction. So render
the *filtered* atom (EWA-style) rather than point-sampling — exact, cheap, and removes a bias
that would otherwise punish high-$\omega$ atoms for an artifact of sampling rather than of the
representation. $\|\omega\\|$ capped at $0.8\pi$. Loss is plain $L^2$; accumulation additive.

## 5. The allocator — instrumental, and itself under test

One global parameter pool $P$. At each growth round, enumerate candidate **moves**:

- **spawn** a blunt atom at a residual hotspot (cost 6): gain estimated by a closed-form
  amplitude solve for a default-shape atom at the hotspot;
- **unlock order $m{+}1$** (cost per §4): gain exact *given everything else frozen* — project
  the residual onto the new basis functions in the atom's frame;
- **unlock $\omega$** (cost per §4): gain estimated from the windowed residual spectrum under
  the atom's envelope; $(c,d)$ by least squares;
- **unlock $\beta$** (cost 1): gain from a 1-D probe along the envelope's radial profile.

Rank by **Δloss per parameter**, execute the top batch, polish, iterate to budget.

**Estimator calibration — required, not optional.** Linear-unlock gains are exact-at-freeze;
spawn and $\omega$ gains are estimates. Ranking exact numbers against estimates biases the
allocator toward whichever estimator is systematically optimistic (plausibly: perpetual
under-spawning). Two safeguards: (a) track the realized/predicted gain ratio per move type
(measured after the following polish interval) and rescale scores by its running median;
(b) a spawn floor — at least 25% of each batch's parameters go to spawns regardless of scores.
The calibration log is a deliverable: it is the direct measurement of how wrong each estimator
is, and doubles as the check on §4's double-counting.

The v1 allocator is deliberately greedy with no lookahead — A2a and the A4 ablations bracket
it, so allocator sophistication is only worth building if the bracket says so.

## 6. Arms

All arms share one growth schedule (init 50% of the parameter pool, add the rest in 4 equal
parameter-batches at fixed iterations, spawn at residual-mass hotspots with the same hotspot
sampler, prune on amplitude modulus with parameters returning to the pool). The arm determines
the *policy*, never the schedule.

**The core triangle** (full grid: every budget, every image, 5 seeds):

| arm | what it is | role |
|---|---|---|
| **A1** | plain Gaussians, $N=P/6$, standard recipe | the incumbent corner |
| **A2a** | fully-unlocked atoms from birth, $\omega$ and $\beta$ set by the same spectral/profile measurements A3 uses, matched $P$ | uniform richness with informed birth — the clean rich corner |
| **A3** | metamorphic + allocator (§5) | the thesis |

**Attribution arms** (mid budget only, all images, 5 seeds):

| arm | what it is | kills/confirms |
|---|---|---|
| **A2b** | fully unlocked, $\omega$ random at birth | the landscape claim: richness + uninformed init should *lose* to A1; predicted highest seed variance of any arm |
| **A4a–c** | A3 with one knob disabled (no $\omega$ / no Hermite / no $\beta$) | which knob carries which class |
| **A4d** | A3 with descended phase: real coefficients plus per-carrier phase offsets $\varphi$, matched cost | **the quadrature claim** — if A4d ≈ A3 in median *and* seed spread, complex coefficients are dead weight and the real-valued literature was right |
| **A5** | A3's final per-atom knob configs **and spawn sites** kept; pose/appearance re-randomized around those sites; standard training, no allocator | path vs architecture: keeps the config–location correspondence so the arm is not a strawman. A5 ≈ A3 ⇒ distill the architecture, drop the curriculum; A5 < A3 ⇒ the homotopy itself matters |
| **A6** | plain Gaussians, spawn density and init scale driven by A3's spectral diagnostics: more, smaller atoms where residual spectral energy is high-frequency, $\sigma \sim 1/\|\omega_{\text{peak}}\|$ | "the measurement did all the work" control, made concrete |

## 7. Targets, budgets, protocol

**Targets** — three content classes, **8 images each** (fewer cannot support a per-class
claim under a paired signed-rank test):

- Kodak, 8 images, 512² center crops, converted to luma (BT.601). *Not* comparable to published
  color full-resolution numbers, and no such comparison will be drawn;
- texture-heavy: 8 crops (grass, fabric, gravel, hair, bark, water, gravel-far, knit; DTD or
  Brodatz);
- cartoon/graphic: 8 flat-color vector-style images.

Grayscale throughout; color is Phase 2 (it adds only accounting disputes — shared vs
per-channel $\omega$ and coefficients).

**Budgets in parameters, not atoms**: $P \in \{6\text{k}, 24\text{k}, 96\text{k}\}$ on 512²
(≈ 44:1, 11:1, 2.7:1 pixels-per-parameter). **The primary cell is $P=24$k.** At 96k all arms
may saturate PSNR and compress differences toward zero (ceiling risk, recorded now); at 6k
nothing may reach acceptable quality on texture. Both remain in the grid as secondary cells and
for the H5 trend, but every headline reading binds at 24k. Report atoms *and* parameters for
every arm.

**Metrics**: PSNR is primary — $L^2$ is the training loss, so it is
the only metric an arm can be *selected* on. SSIM and LPIPS are reported, never selected on;
a PSNR win that inverts under LPIPS is reported as exactly that. Efficiency: iterations and
wall-clock to reach within 0.5 dB of the arm's own final PSNR (per-class absolute thresholds
are meaningless: unreachable for texture at 6k, trivial for cartoons). Robustness: across-seed
IQR per cell, a primary metric with its own reading (H6).

**Protocol:**

- Adam, per-group learning rates. Every arm gets an identical LR sweep (3 values × the same two
  pre-named groups: position LR and appearance LR) on 2 held-out tuning images. Equal tuning
  budget per arm — the incumbent's folk-tuning advantage is a confound otherwise.
- 20k iterations; convergence check at the end (loss slope over the last 2k must be < 1% of
  total decrease, else the cell is flagged and extended once by 10k — flagged cells are
  reported as such).
- **5 seeds** per (arm × image × budget in that arm's matrix).
- Statistics: per-image paired differences against A1, Wilcoxon signed-rank across the 8 images
  of a class (and across all 24 for pooled statements), medians with IQR. No means of PSNRs.
- Sanity phase before the grid: planted-atom recovery, one test per knob, from each arm's own
  init policy. A2b is expected to fail high-$\omega$ recovery; that predicted failure validates
  the landscape mechanism cheaply before the GPU-days.

## 8. Pre-registered readings

Fixed before running; a null is a null. All headline readings bind at the primary cell
($P=24$k) unless stated.

- **H1 (headline).** A3 > A1 on **all three** classes — the family, unlike a Gabor-only
  enrichment, covers edges and regions too, so a texture-only win is a *failure of the family
  framing* (it collapses to "Gaussians + one texture knob"). Margins: ≥1 dB class-median on
  texture, ≥0.3 dB on Kodak and cartoon.
- **H2.** A3 > A2a: adaptivity beats uniform richness *with birth policy held equal* — A2a, not
  A2b, is the comparator, else richness is confounded with init. If A2a ≥ A3, the allocator is
  dead weight: unlock everything, skip §5.
- **H3 (attribution, exploratory).** The spending map is content-locked: $\omega$ unlocks
  concentrate on texture, Hermite on edges/junctions, $\beta$ on region boundaries. Deliverable:
  knob-type overlay maps. Explicitly exploratory — no threshold, and a suggestive map is a
  hypothesis for a later experiment, not a finding of this one.
- **H4.** A5 < A3 confirms path-dependence; A5 ≈ A3 is the useful negative — distill the
  architecture, drop the curriculum.
- **H5.** The A3−A1 gap is larger at 6k than at 96k (richness matters most when atoms are
  scarce). Trend read across the full grid; ceiling-flagged cells excluded.
- **H6 (seed variance).** IQR ordering: A2b > A4d > A3 ≈ A2a ≈ A1. The landscape story in one
  line — descended nonlinear parameters produce spread, informed birth and linear enrichment
  remove it. If A4d matches A3's IQR, the quadrature trick buys nothing measurable.
- **H7 (quadrature, median).** A3 ≥ A4d in class medians. H6+H7 together are the test of the
  one design principle this document actually invented.
- **H8.** A6 closes less than half of the A3−A1 gap. If it closes most of it, the result is a
  spawn heuristic for plain Gaussians — publish that instead.
- **Kill condition.** At the primary cell, at matched $P$ *and* matched wall-clock: A1 ≥ A2a
  and A1 ≥ A3 on every class → the richer-family program loses to "just add more Gaussians" at
  these budgets, and count is the only knob that matters below ~100k parameters.

## 9. Out of scope, deliberately

Family members that change the *algebra* rather than the atom — occlusion/`over` compositing,
shared orientation fields and domain warps, gradient-domain synthesis, noise-carrying texture
atoms under a distributional loss, fractal self-reference. They do not fit a matched-parameter
atom-family comparison (they alter the synthesis operator, so parameters-per-atom stops being
the right axis) and each needs its own control structure. Second experiment, if this one earns
it.

Also out of scope for Phase 1: rate. Parameters are not bits — a frequency plausibly needs more
precision than a position — so a params-matched win can still die at matched bits. Phase 2
quantizes per parameter type (fp16 / 8-bit per-group) and plots rate–distortion; the per-atom
knob mask (a few bits per atom) is charged to the representation there.

## 10. Cost

Run matrix: core triangle 3 arms × 24 images × 3 budgets × 5 seeds = 1,080 fits; attribution
arms 7 × 24 × 1 × 5 = 840; sanity + tuning ≈ 100. **≈ 2,000 fits** at ~2–4 min each on one
modern GPU → **4–5 GPU-days**, embarrassingly parallel. Implementation ~1,400 lines of PyTorch;
the nonstandard machinery is §5's move scoring (windowed-FFT measurement, closed-form
projections, calibration log) and §4's analytic pixel filter. Recommended first cut: sanity
phase plus the core triangle on four images at the primary budget — an afternoon-scale run that
decides whether the full grid is worth its GPU-days.
