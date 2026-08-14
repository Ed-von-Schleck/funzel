# Metamorphic splats: can richer primitives beat more Gaussians?

An experiment design. Self-contained and independent of the documents in `docs/` — it shares no
premises with them and none of their conclusions are assumed here.

**Status: planned, not run.** Nothing below is a result.

---

## 1. The question

2D Gaussian splatting represents an image as $\sum_i c_i\,G(\mu_i,\Sigma_i)$ — many cheap
primitives, 6 parameters each, fit by random init + Adam. The question: at a **fixed total
parameter budget** $P = N \times k$, is it better to spend $P$ on more atoms ($N\uparrow$,
$k=6$) or on richer atoms ($k\uparrow$, $N\downarrow$)?

Framing that makes the question non-trivial: the curve from $(N{=}10^6, k{=}3)$ — pixels — to
$(N{=}1, k{=}10^6)$ — an implicit neural representation — passes through Gaussian splatting near
the cheap end. Whether the distortion-at-budget curve dips in the middle is content-dependent:
an edge wants many cheap anisotropic atoms, a texture patch wants one rich oscillatory one, a
sky wants one enormous smooth one. So the object under test is not a single richer primitive
but an **allocation policy** that decides, per location, where on the $N$-vs-$k$ curve to sit.

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
- **Every atom is born a Gaussian.** Each enrichment gets a blunting value at which the atom is
  a plain Gaussian, and moves off it only when the gradient (or the allocator) demands.
- **Domain knowledge enters at birth, not during descent.** Nonlinear parameters that cannot be
  found by descent (a carrier frequency, by the uncertainty principle: capture radius in space
  $\sigma$ times capture radius in frequency $1/\sigma$ is pinned at $\sim 1$, so position and
  frequency cannot both have wide basins) are set by a matched-filter measurement of the
  residual at spawn time. Adam polishes; birth places.

## 3. Prior art, briefly

The kernel-swap literature is active but fragmented; none of it tests the allocation question.

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
  uses complex amplitude as an *optimization device* for ordinary photographs.
- **Other kernel swaps**: GES's learnable exponent
  ([arXiv:2402.10128](https://arxiv.org/abs/2402.10128)), Gaussian–Hermite splatting
  ([arXiv:2408.16982](https://arxiv.org/abs/2408.16982)) — which needed a coarse-to-fine
  schedule on Hermite order, consistent with §2's basin-narrowing prediction — Student-t, Beta,
  and convex kernels. Each swaps one kernel uniformly; none allocates richness adaptively.

## 4. The unified atom

$$f(x) \;=\; \mathrm{Re}\!\left[\,P_c(u)\;e^{i\,\omega^\top u}\right]\cdot e^{-\frac12\|u\|^{\beta}},
\qquad u = A(x-\mu)$$

with $A$ the usual anisotropic pose (2 log-scales + rotation), $\mu$ position, and $P_c$ a
**complex-coefficient** polynomial in the Hermite basis of order $m$. Three knobs, each with a
blunt zero; at all-blunt the atom is exactly a plain Gaussian with 6 parameters (grayscale).

| knob | blunt value | unlocked cost | serves | sieve status |
|---|---|---|---|---|
| polynomial order $m$ | $m=0$ | +2 per complex coeff | edges, ridges, junctions | linear ⇒ free for the optimizer; marginal gain exact by projection |
| carrier $\omega$ | $\omega=0$ | +2 | periodic texture | nonlinear ⇒ set by spectrogram birth (§2) |
| envelope exponent $\beta$ | $\beta=2$ | +1 | region boundaries, flat fills | nonlinear but demonstrated trainable (GES) |

Complex coefficients generalize the quadrature trick: phase never becomes a descended parameter
at any order — the appearance half of the atom stays linear given pose, $\omega$, $\beta$.
Hermite order and carrier are complementary, not redundant: order $m$ buys $m$ structured
wiggles for $O(m)$ parameters (right for edges); $\omega$ buys unlimited periodic cycles for 2
(right for texture). That division of labor is itself a testable prediction (H3).

$|\omega|$ is capped at Nyquist. Rendering is additive accumulation; loss is plain $L^2$.

## 5. The allocator — the object under test

One global parameter pool $P$. At each growth round, enumerate candidate **moves**:

- **spawn** a blunt atom at a residual hotspot (cost 6);
- **unlock order $m{+}1$** on an existing atom (cost ~4 at order 1): marginal gain is exact —
  project the residual onto the new basis functions in the atom's frame;
- **unlock $\omega$** (cost 2, amplitudes retuned): gain estimated from the windowed residual
  spectrum under the atom's envelope; $(a,b)$ by least squares;
- **unlock $\beta$** (cost 1): gain from a 1-D probe along the envelope's radial profile.

Rank all moves by **Δloss per parameter**, execute the top batch, iterate to the budget. This
is matching pursuit over a move space that contains "sharpen an existing atom" alongside "add
an atom"; the most common move's gain is computed exactly rather than guessed, which is what
the linearity principle buys. The v1 allocator is deliberately greedy with no lookahead — arms
A2 and A4 bracket it, so allocator sophistication can come later if the bracket says it is
worth having.

## 6. Arms

All arms share one growth schedule (init 50% of the budget, add the rest in 4 equal batches at
fixed iterations, spawn at residual-mass hotspots with the same hotspot sampler, prune on
amplitude modulus and redistribute). The arm determines the *policy*, never the schedule.

| arm | what it is | kills/confirms |
|---|---|---|
| **A1** | plain Gaussians, $N=P/6$, standard recipe | the incumbent corner |
| **A2** | fully-unlocked atoms from birth, matched $P$ | richness *without* adaptivity |
| **A3** | metamorphic + allocator (§5) | the thesis |
| **A4a–c** | A3 with one knob disabled (no $\omega$ / no Hermite / no $\beta$) | attribution |
| **A5** | A3's final per-atom configuration, retrained from scratch, random init | path vs architecture: if A5 ≈ A3, metamorphosis is just architecture search and can be distilled; if A5 < A3, the homotopy matters |
| **A6** | plain Gaussians, spawn placement using A3's full diagnostics | "the birth heuristic did all the work" control |

## 7. Targets, budgets, protocol

**Targets** — three content classes, because every prediction is class-conditional:

- Kodak, 8 images, 512² center crops (realism anchor, comparable to published numbers);
- texture-heavy: 4 crops (grass / fabric / gravel / hair, DTD or Brodatz);
- cartoon/graphic: 4 flat-color vector-style images.

Grayscale throughout; color is Phase 2 (it adds only accounting disputes — shared vs per-channel
$\omega$ and coefficients).

**Budgets in parameters, not atoms** — the single most common way to fool yourself here:
$P \in \{6\text{k}, 24\text{k}, 96\text{k}\}$. Report atoms *and* parameters for every arm.

**Protocol:**

- Adam, per-group learning rates. Every arm gets an identical small LR sweep (3 values on its
  two most sensitive groups) on 2 held-out tuning images. Equal tuning budget per arm — the
  incumbent's folk-tuning advantage is a confound otherwise.
- 20k iterations fixed; also record iterations-to-threshold (PSNR 30) and wall-clock. Report at
  iteration parity *and* wall-clock parity: a rich atom costs more FLOPs but there are fewer,
  and whether that washes is measured, not asserted.
- **5 seeds** per (arm × image × budget). Seed variance is a primary metric, not noise: the
  landscape argument predicts descended-nonlinear arms have high across-seed spread and
  linear/birth arms low.
- Statistics: per-image paired differences, Wilcoxon signed-rank across the 16 images, medians
  with IQR. No means of PSNRs across images.
- Sanity phase before the grid: planted-atom recovery, one test per knob (can the arm recover a
  single planted rich atom from its own init policy?). A2-style random init is expected to fail
  the high-$\omega$ recovery; that failure validates the mechanism cheaply before the GPU-days.

## 8. Pre-registered readings

Fixed before running; a null is a null.

- **H1 (headline).** A3 > A1 at matched $P$ on **all three** classes — the family, unlike a
  Gabor-only enrichment, covers edges and regions too, so a texture-only win is a *failure* of
  the family framing (it would mean the family collapses to "Gaussians + one texture knob").
  Margin to care about: ≥1 dB median on texture, ≥0.3 dB elsewhere.
- **H2.** A3 > A2: adaptivity beats uniform richness. If A2 ≥ A3, the allocator is dead weight —
  unlock everything and skip §5.
- **H3 (attribution).** The spending map is content-locked: $\omega$ unlocks concentrate on
  texture, Hermite on edges/junctions, $\beta$ on region boundaries. Deliverable: knob-type
  overlay maps per image. If the spending map reads as an unsupervised structure segmentation,
  that is a result in its own right.
- **H4.** A5 < A3 confirms path-dependence; A5 ≈ A3 is the useful negative — distill the
  architecture, drop the curriculum.
- **H5.** A6 closes less than half the A3−A1 gap. If it closes most of it, publish the birth
  heuristic and not the primitive.
- **Kill condition.** A1 ≥ {A2, A3} on every class at matched $P$ *and* matched wall-clock:
  at these budgets the richer-family program loses to "just add more Gaussians," and count is
  the only knob that matters below ~100k parameters.

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

6 arm-families × 16 images × 3 budgets × 5 seeds ≈ 1,700 fits at ~2–4 min each on one modern
GPU → **2–4 GPU-days**, embarrassingly parallel. Implementation ~1,200 lines of PyTorch; the
only nonstandard machinery is the move-scoring of §5 (windowed-FFT birth sampler, closed-form
projections). Recommended first cut: sanity phase plus A1/A2/A3 on four images at one budget —
an afternoon-scale run that decides whether the full grid is worth its GPU-days.
