# Optimal encoding of images as 2D Gaussians

Given a target image, find the best representation as a set of 2D Gaussians, and know how far
from best any given representation is.

Not a compression document — rate, entropy coding and decode speed are out of scope.

**Status tags.** **[V]** verified against a primary source in `papers/` · **[V\*]** verified
with an inline caveat · **[U]** unverified, search summaries only · **[A]** analysis from
**[V]** facts · **[A←U]** analysis resting on an unverified premise. Methodology rules and this
document's error history are in Appendix A.

---

## 1. Summary

**The central negative result (§10).** On all six certified rows, **greedy** produced a better
$N$-atom encoding than the convex BLASSO route, even after debiasing and full polish. Adding an
$\ell_1$ term buys a certificate and costs reconstruction quality.

*Terminology:* "greedy" here means **certificate-driven greedy at $\lambda=0$** — it places each
new atom at $\arg\max_\theta|\eta(\theta)|$, so it uses the dual certificate as a *placement
criterion*, but it solves no convex problem and yields **no optimality certificate**. It is also
roughly an order of magnitude cheaper than the BLASSO route, which repeats a comparable inner
loop across a $\lambda$ bisection. This is matching pursuit with continuous refinement.

**What that means for the framing.** (P$\lambda$) is *not* justified as the working definition
of optimal encoding on quality grounds. Its defensible role is narrower: a **diagnostic
instrument** telling you where a solution sits relative to a computable optimum, not a method
for producing the best solution.

**What survived scrutiny:**

| | status |
|---|---|
| Greedy beats random restarts | **[V]** 9/9 rows, §10.1 |
| Greedy beats BLASSO at matched $N$ | **[V]** 6/6 *certified* rows, §10 (3 more agree but BLASSO did not converge there) |
| How far greedy is from optimal | **not bounded by anything here** — §11 U14 |
| The optimality certificate is computable and self-diagnosing | **[V]** §6, §9.4 |
| TV needs norm-weighting over a scale-varying dictionary | **[V]** §7.1.1 |
| Forward model linear, loss $L^2$ — preconditions hold | **[V]** §3 |
| Relaxation gap non-monotone in separation | **[V]** §9.1; mechanism untested |
| Curvelet-optimal rates are *achievable* by the dictionary (existence) | **[V]** §4; **never tested empirically** |

**What is not established.** Everything measured is at $N\le16$ atoms on $56^2$–$192^2$ images.
Real encoding is $10^3$–$10^5$ atoms on $10^6$ pixels — three to four orders of magnitude away.
No conclusion below is known to survive that gap, and §9.3 shows the solver cannot currently
reach it. §11 lists every open question with the experiment that would settle it.

---

## 2. Problem statement

**(P0) Best $N$-term approximation.**

$$\min_{\{c_i,\mu_i,\Sigma_i\}_{i=1}^N} \Big\| I - \sum_{i=1}^N c_i\, G(\mu_i,\Sigma_i) \Big\|^2$$

What "optimal encoding" intuitively means. Intractable in the worst case (§5).

**(P$\lambda$) Rate-regularized encoding over measures.** With $\Theta$ the atom parameter
space and $m$ a Radon measure on it:

$$\min_{m \in \mathcal{M}(\Theta)} \tfrac12 \|\Phi m - I\|^2 + \lambda |m|(\Theta), \qquad \Phi m = \int_\Theta \varphi_\theta \, dm(\theta)$$

**[V]** The BLASSO, exactly as in `papers/1811.06416v1.pdf`. $|m|(\Theta)$ is the total
variation norm; for $m=\sum_i c_i\delta_{\theta_i}$ it equals $\sum_i|c_i|$. Convex,
infinite-dimensional. $N$ emerges from $\lambda$ rather than being fixed.

**[A]** The absence of permutation symmetry holds for the *formulation*. Every algorithm
discretizes into labelled particles, restoring the $N!$ symmetry in the optimized landscape.

**(P∞) Achievable rate.** How fast can error decay with $N$ for a given image class? (§4)

### 2.1 The relaxation gap

(P$\lambda$) is the continuum $\ell_1$ relaxation of $\ell_0$. Measured (§9):

- in-model, dense tiling: **0.4%** (exact)
- in-model, intermediate separation: **~22%** (exact)
- out-of-model (photographs, cartoon): **1.0–4.8%** (lower bound)

**[A]** In-model, the gap is small at dense-tiling separation, so certifying (P$\lambda$)
there is informative about (P0). Out-of-model the figures are **lower bounds**: the true gap is
*at least* 1.0–4.8% and could be far larger, so no upper bound on the cost of certification is
established for real images. §11 U7.

**[A]** And §10 shows the cost is not worth paying if the goal is the best encoding, because a
method with no certificate does better.

---

## 3. Preconditions

### 3.1 Linearity of the forward model — **holds**

**[V]** `papers/2403.08551v5.pdf` §3.2. GaussianImage folds accumulated transparency into
opacity, then merges colour and opacity into one learnable coefficient:

$$C_i = \sum_{n\in N} c'_n \exp(-\sigma_n), \qquad \sigma_n = \tfrac12 d_n^T\Sigma^{-1}d_n \tag{Eq. 7}$$

No sorting, no transparency, no separate opacity, no normalization by accumulated weight.
**[A]** This is exactly $\Phi m$, linear in $c'_n$.

**[A] Scope:** verified from the rendering equation, not the released CUDA kernel. An output
clamp to $[0,1]$ before the loss would break linearity at saturation without appearing in
Eq. 7. **Unresolved — §11 U1.**

**[A]** The atom family is identical to §4's: $A^TA=\tfrac12\Sigma^{-1}$ ranges over all SPD
matrices as $A$ ranges over $GL(2,\mathbb{R})$.

**[A]** 3D Gaussian splatting does **not** satisfy this — $\alpha$-blending is nonlinear
through occlusion. Nothing here transfers to 3D.

### 3.2 The loss must be Hilbertian — **holds for the baseline**

The certificate is $\eta_\lambda=\Phi^*p_\lambda$ and needs an adjoint. $L^2$ works; SSIM and
L1 do not. **[V]** GaussianImage's representation objective is $L^2$.

**[A]** Still a restriction on scope: **[U]** Image-GS optimizes a perceptual objective, and
any method with SSIM or L1 terms admits no certificate. Whether $L^2$-optimal encodings are
perceptually acceptable at low atom counts is outside what this framework can answer — §11 U9.

### 3.3 Colour — **unresolved**

Theory is scalar-valued; the problem is three channels sharing one geometry. Independent
per-channel BLASSO loses the shared geometry. **[U]** Group BLASSO (amplitudes as 3-vectors,
penalty $\sum_i\|c_i\|_2$) preserves it and exists in the literature, unverified for this case.
**[A←U]** The certificate should generalize to $\|\eta(\theta)\|_2\le1$. §11 U2.

---

## 4. What the dictionary can achieve — theory, untested here

**[V]** `papers/1910.10319v1.pdf` — Erb, Hangelbroek & Ron. The dictionary studied is the
splatting dictionary: $\mathcal{D}=\{\varphi\circ M \mid M\in GL(d,\mathbb{R})\ltimes\mathbb{R}^d\}$,
full anisotropy including rotation.

**Theorem.** For $f$ in the Donoho–Candès cartoon class ($f_1+f_2\chi_\Omega$, both $C^2$,
$\Omega$ with $C^2$ boundary), Gaussian mixtures attain $O(N^{-1}(\log N)^{3/2})$ — the same
rate as curvelet $N$-term thresholding.

### 4.1 What this does and does not say

**Non-circular for the cartoon class.** That class is defined geometrically and its curvelet
decay is a cited theorem, so this is a genuine statement about an independently defined class.
(The companion $\mathcal{C}^\alpha$ result *is* partly circular — $\mathcal{C}^\alpha$ is
defined by curvelet decay.)

**Four limits.**

1. **Asymptotic, constant $C_f$ unquantified.** At $N=10^3$–$10^5$ this constrains little.
2. **[V\*] Non-adaptive construction.** Sub-budgets are independent of $f$; the geometry comes
   from the curvelet tiling, only coefficients are data-dependent. An achievability result for
   the dictionary, not an encoder.
3. **[A]** It bounds (P0) with unconstrained amplitudes. (P$\lambda$) additionally penalizes
   $\sum_i|c_i|$, so the rate does not transfer without amplitude control the theorem does not
   provide. It is stated on $L^2(\mathbb{R}^2)$, not a pixel grid.
4. **[A] — audit (M3).** It does **not** imply near-optimal solutions look curvelet-like. The
   theorem exhibits *one* construction; with a redundant dictionary there is no reason to
   expect structural uniqueness. An earlier version made that converse error.

**[A] Defensible statement:** *the dictionary is not asymptotically limiting for cartoon-like
images.* Nothing stronger. **No empirical contact** — §11 U6 gives the test.

---

## 5. Why (P0) is hard

- **[U]** Sparse approximation over a general dictionary is NP-hard, and NP-hard to approximate
  within any factor, including under *coherent* dictionaries; the Gaussian dictionary is highly
  coherent.
  **[A] — audit (M3):** worst-case hardness over a family says nothing about the instances
  arising here. The §9.5 diagnostic found greedy recovering an in-model target *exactly*
  (0.0000%) at wide separation — a family of easy instances.
- The dictionary is continuous: an infinite-dimensional non-convex search.
- **[A]** (P0) has $N!$ equivalent global minima and symmetry-induced saddles.
- **[U]** SteepGS reports stuck primitives sit at *saddle points*, not local minima.

---

## 6. The optimality certificate

**[V]** With $p_\lambda=\tfrac1\lambda(I-\Phi m_\lambda)$ and $\eta_\lambda=\Phi^*p_\lambda$,
optimality is characterized by $\|\eta_\lambda\|_{\infty,\Theta}\le1$.

$\eta_\lambda$ is the residual correlated against **every candidate atom** — a function over
$(\mu,\Sigma)$, not over pixels, so its argmax specifies a position *and* a shape.

**[A]** Over a scale-varying dictionary this must be the correlation against **unit-norm**
atoms; using the raw inner product makes the argmax prefer wide atoms and is what §7.1.1
measures. The two coincide only when all atoms share a norm, which is the classical
translation-only setting.

**Two properties, both from convex duality**, needing no separation, kernel condition or
non-degeneracy:

1. a global optimality test for (P$\lambda$);
2. **[V]** a suboptimality bound at every iteration, via the Frank-Wolfe rate
   $T_\lambda(m^{[k]})-T_\lambda(m^\star)\le C_1/k$ and the duality gap.

### 6.1 What it costs

**[A] It needs compactness too.** The test is a supremum over $\Theta$, so it must be attained.
**[V]** SFW assumes exactly this, maximizing $|\eta|$ *"over the compact domain $X$"*.

| | requires | truncating the scale range gives |
|---|---|---|
| certificate | $\Theta$ compact, $\eta$ continuous | ✔ |
| CPGD (A1) | compact **and boundaryless** | ✘ boundary introduced |

So the certificate certifies (P$\lambda$) **restricted to the truncated $\Theta$**. Any
reported gap must state the truncation.

**[V] It does not scale in the current implementation** — §9.3. Certified at $N\le16$ and
uncertified at $N=32$ on all three targets; **[A]** the boundary between was not located, and
nothing was tested above 32.

**[A←U] Relation to practice.** Densification heuristics reported for other methods —
pixel-space error sampling, distortion-driven growth, positional-gradient magnitude — look like
surrogates for $\eta_\lambda$ that do not search the covariance dimension. The descriptions come
from search summaries, not from the papers or code, so this is suggestive only. **[V]**
GaussianImage itself has no densification at all.

---

## 7. Which imported guarantees transfer

**[V] SFW (`1811.06416v1.pdf`).** Algorithm: argmax $|\eta|$ → LASSO on amplitudes → **joint**
non-convex slide → prune. Theorem 3 gives finite termination under uniqueness and
non-degeneracy ($|\eta_\lambda|<1$ off support, $\eta_\lambda''(\theta_i)\neq0$ on it).

**[V\*]** Proved for $d=1$. **[V] The dictionary is a fixed kernel with translation as the only
parameter** — KER$(k)$'s worked example is $\phi(x)=e^{-(\cdot-x)^2}$ on $[0,1]$. So the
"$d\in\mathbb{N}^*$" remark concerns the dimension of *position*, not free covariance. **[A]**
KER(2) also fails at both ends of scale for free covariance.

### 7.1 Conic Particle Gradient Descent

**[V]** `1907.10300v2.pdf`. Particle gradient descent on positions and weights; $\alpha$ =
weight step, $\beta$ = position step. For $\alpha,\beta>0$ this is gradient flow in the
**Wasserstein–Fisher–Rao** metric, weights updating multiplicatively. Theorem 4.2 gives global
convergence under (A1–5) with $W_\infty(\nu_0,\rho)$ small and $\beta/\alpha$ small.

| | statement **[V]** | verdict for $\Theta=\mathbb{R}^2\times\mathrm{SPD}(2)$ |
|---|---|---|
| A1 | compact Riemannian manifold **without boundary**; $\varphi,R$ twice Fréchet differentiable | **fails as stated** |
| A2 | $h(r)=r^2$, metric $(\alpha,\beta/r^2)$ | holds — a choice of algorithm |
| A3 | $R$ convex | **holds** (square loss) |
| A4 | unique, finitely supported minimizer | not checkable a priori |
| A5 | $\nabla^2R\succ0$, curvature, **strict slackness** | not checkable a priori |

**[A] — audit (M3).** A1 fails as stated: the space is non-compact, and bounding it introduces
a boundary A1 also excludes. Positions compactify to a torus and orientation is $S^1$;
log-scale has no natural boundaryless compactification. **But this withdraws the guarantee; it
does not show CPGD converges badly here** — nothing tests that. Boundarylessness may be a
convenience of the gradient-flow argument: if the optimum's scales lie strictly interior, the
boundary is never active. **Unresolved — §11 U3.**

**[V] The author's own position on A5:** *"depend on an a priori unknown object… it is an open
question to even show local convergence when (A5) does not hold."*

**[V] Regularization is required for the rate.** $\lambda>0$ is necessary for (A5) in the
signed case, and $\kappa_0=O(\lambda)$ — as $\lambda\to0$ the exponential rate is lost.
**[A]** A genuine tension: small $\lambda$ shrinks the relaxation gap, large $\lambda$ buys the
rate.

#### 7.1.1 The TV norm is not commensurate across scales

**[A]** Every atom in a translation-only dictionary has the same norm, so $\lambda\sum_i|c_i|$
prices them alike. With free covariance $\|\varphi_\theta\|^2=n^2\pi e^{-(u+w)}$ grows with
width, so a wide atom buys more inner product per unit penalty.

**[V] Measured** (5px atoms, 192px image): at a *true* atom centre the raw
$\langle\varphi,y\rangle$ rises monotonically with candidate width, 18.8 (2.5px) → 105.0
(80px). At a location with **no atom**, an 80px candidate scores 141.2 — higher than anything
at a true centre. Normalized by $\|\varphi_\theta\|$ the scan peaks exactly at the true width.

**[V] Decisive.** Unnormalized, greedy placed its first atom 57px from any true centre.
Normalized, it recovered all four atoms to 0.00px and exactly 0.0000% error.

**[A] — audit.** This is **not a new theoretical result.** Normalized dictionaries are a
standard convention throughout sparse approximation (MP, OMP, LASSO all assume it), and
**[V]** `papers/tip2006.pdf` specifies unit-$L^2$-norm generating functions. The honest
statement: *a standard convention was violated here, and this is the measured cost.* The
commensurate regularizer is $\sum_i|c_i|\,\|\varphi_{\theta_i}\|$.

### 7.2 Other results

**[V] Semi-discrete OT** (`ma.pdf`, `dGBOD12.pdf`): damped Newton with global linear
convergence for matching a **prescribed density**, via concave maximization over power-diagram
weights. **[A]** Rigorous about density matching, but the relation of that objective to
reconstruction error is **unestablished** — so this is a *candidate* initializer of untested
value, not a principled one and not a solved subproblem of (P0). §11 U10 is the test, and if it
comes back negative the guarantee is exact about something irrelevant.

**[V] Hessian anisotropy** (`BLdG+16.pdf`): anisotropy conforming to the Hessian of a **convex**
function, for any prescribed density. **[V\*]** Image Hessians are indefinite; the standard
repair is $|H|$ or the structure tensor, and whether guarantees survive it is not established.

**[V] EM precedent** (`gbc.pdf`, Goal-Based Caustics 2011): CCVT initialization → anisotropic
Gaussian kernels → sparse EM, ~1024 kernels. They tested several initializations and *"obtained
the most reliable results using regular point sets that adapt to the image intensity."* No
optimality claim; EM converges to a local optimum only.

**[U] SteepGS**: minimal offspring count 2, displacement along the minimal eigenvector. Derived
for the 3DGS objective *with* $\alpha$-blending — the case §3.1 says the measure theory does
not cover.

### 7.3 The guarantees do not compose

| result | guarantees | for which problem | guarantee transfers? |
|---|---|---|---|
| BLASSO certificate | global optimality test | any bounded linear $\Phi$, $L^2$, scalar, **$\Theta$ compact** (§6.1) | **yes**, on the truncated $\Theta$ |
| FW duality gap | suboptimality bound | any bounded linear $\Phi$ | **yes** |
| SFW Thm 3 | finite termination | $d=1$, fixed kernel, translation-only | no |
| CPGD Thm 4.2 | global convergence | (A1–5) | no — A1 fails as stated |
| KMT Newton | global linear convergence | density matching, not (P0) | different objective |
| SteepGS split | optimal offspring count | 3DGS with $\alpha$-blending | different objective |

**[A] — audit (M3):** this reports whether the *guarantee* carries over, not whether the
guaranteed property holds. §9.1 is the cautionary case — the recovery guarantee does not
transfer at image densities and recovery was exact anyway far below the threshold.

---

## 8. The regime mismatch

**[V]** The recovery theory assumes sparse, well-separated atoms. `2509.12889v5.pdf` Theorem
5.1 requires separation in a semi-distance normalizing centre gaps by combined width. **[A]**
Evaluated for $d=2$: $\Delta\approx9$–10.5 almost independently of atom count, giving a
required semi-distance of 18–21, i.e. **centre separation of ~28–35 widths**. A dense image
tiling gives 1–2.

**[A] — audit (M3).** This is a *sufficient* condition. Failing it withdraws guarantees; it
does not establish that recovery fails. §9.1 measured recovery **exact at 8 widths** and broken
at 4, so the true breakdown lies in $(4,8]$ against a stated 28–35 — an overstatement by a
factor between **3.5 and 9**, consistent with the constants (11.9, 0.3025) being proof
artefacts. The breakdown was bracketed, not located.

**No longer guaranteed** at image densities: exact support recovery, SFW finite termination,
(A4) uniqueness, (A5) strict slackness. **What is measured:** §9.

---

## 9. Experimental results

`experiments/e2_relaxation_gap.py` (in-model targets, exact optimum) and
`experiments/e2b_natural.py` (out-of-model targets, bounded optimum).

**Scale caveat, applying to everything below.** Certified rows are $N\le16$ atoms on
$56^2$–$192^2$ images (§9.3 is why). Real encoding is $10^3$–$10^5$ atoms on $10^6$ pixels. No
result here is known to survive that. Statistical support is two instances per point in §9.1
and one in §9.2, with no error bars.

### 9.1 In-model targets: the gap is non-monotone in separation

Targets are exact $K$-atom mixtures, so the (P0) optimum is **exactly 0** and the gap is the
BLASSO error itself. Fixed width in pixels, swept centre spacing; unit-norm dictionary; $L^2$;
single channel; truncated $\Theta$.

| $r$ (widths) | 30 | 15 | 8 | 4 | 2 | 1 |
|---|---|---|---|---|---|---|
| BL debiased, run A | 0.00 | 0.00 | 0.00 | 22.86 | 4.70 | 0.38 |
| BL debiased, run B | 0.00 | 0.00 | 0.00 | 21.52 | 4.01 | 0.42 |
| absolute error, A | 0 | 0 | 0 | 35853 | 18677 | 1711 |

Percentages of $\tfrac12\|y\|^2$; certificate confirmed convergence throughout
($\le0.04\%$). **[A]** The two runs share a random stream that diverges once the restart counts
differ, so $r=30$ is the *same* instance computed twice (a reproducibility check) while $r\le15$
are genuinely independent instances.

**[V] Exact support recovery down to $r=8$**, then a sharp transition, then partial recovery.
Absolute error falls alongside the percentage from $r=4$ to $r=1$, so this is not the
normalization effect of §9.5.

**[A] Hypothesis, not finding (M2).** A natural reading is that recovery difficulty and
approximation difficulty peak at different separations: wide separation permits recovery; heavy
overlap makes the representation redundant, so many atom sets reconstruct well and
approximation is easy *because* recovery is hopeless; the middle has neither. **No experiment
here isolates this mechanism**, and competing explanations — $\lambda$-matching behaving
differently across densities, or $K_{BL}$ drifting from target (3–10 against targets of 4) —
are not excluded. §11 U4.

### 9.2 Out-of-model targets: photographs and a cartoon

No exact representation, so no known optimum. The reference is the best of certificate-driven
greedy at $\lambda=0$, random restarts, and the polished BLASSO solution (itself a feasible
$N$-atom point). Since $E_{\rm ref}\ge E_{\rm opt}$, the penalty is a **lower bound** on the
relaxation gap.

| target | $N$ | $E_{\rm ref}$ | BL debiased | **BL polished** | penalty | certgap | $r_{\rm emp}$ | $r_{\rm ref}$ |
|---|---|---|---|---|---|---|---|---|
| cartoon | 8 | 2.10 | 6.17 | 2.50 | 4.08 | 0.003 | 0.32 | 1.30 |
| cartoon | 16 | 1.15 | 2.16 | 1.20 | 1.01 | 0.004 | 0.74 | 1.75 |
| ascent | 8 | 8.30 | 13.14 | 9.59 | 4.83 | 0.024 | 0.94 | 2.52 |
| ascent | 16 | 5.79 | 9.12 | 6.67 | 3.33 | 0.004 | 0.91 | 2.27 |
| face | 8 | 5.25 | 7.85 | 5.72 | 2.60 | 0.004 | 2.07 | 1.97 |
| face | 16 | 3.59 | 5.53 | 4.29 | 1.94 | 0.002 | 2.02 | 2.94 |

$r_{\rm emp}$/$r_{\rm ref}$: empirical separation ratios (median nearest-neighbour centre
distance / median fitted width) for the BLASSO and reference solutions. $N=32$ rows excluded as
uncertified (§9.3).

1. **Penalty 1.0–4.8%**, falling with budget, on every target. A lower bound.
2. **[A] Withdrawn: "out-of-model costs more than in-model" is not supported.** An earlier
   version compared 1.0–4.8% here against an *exact 0.4%* in-model — but that in-model figure is
   at $r=1$, whereas these fits sit at $r_{\rm ref}\approx1.3$–2.9. At matched separation
   ($r\approx2$) §9.1 gives 4.0–4.7%, which **overlaps** the out-of-model range. Comparing
   against $r=1$ selected §9.1's smallest value. Nothing here separates the model-class effect
   from the separation effect — that is §11 U12.
3. **Real fits land at $r_{\rm ref}\approx1.3$–2.9**, between §9.1's easy point and its peak.
   **[A]** Interpolating §9.1 across that range predicts 0.4–10%, and the observed 1.0–4.8% sits
   inside — but a 25-fold prediction interval is close to unfalsifiable, so this is consistency,
   not corroboration.
4. **[A] Hypothesis (M2).** The cartoon is best-behaved (1.01% at $N=16$ against 1.94 and
   3.33), which would mean §4's theory applies most cleanly to the target least like a
   photograph. Three targets, one instance each.
5. **[A] Not established.** BLASSO clusters relative to the reference on cartoon and ascent but
   not on `face` at $N=8$. Target-dependent; mechanism unknown.

### 9.3 The certification ceiling is an implementation limit

**[V]** certgap is 0.002–0.024% at $N=8,16$ on all three targets, then 1.5–6.5% at $N=32$.

**[V] Not intrinsic.** Holding $\lambda$ fixed at ~35 atoms and varying only inner iterations
gave 0.83% / 1.06% / 7.24% at 400 / 1200 / 3000 — the gap gets *worse* with more optimization.
**[A]** Better inner solves zero more amplitudes, the prune threshold removes them, and
Frank-Wolfe churns until its iteration budget is exhausted without satisfying
$\sup|\eta|\le\lambda$.

**[A]** So §6 gains no scaling limitation — but this is the binding engineering obstacle. §11 U5.

### 9.4 What the certificate did

Three times, and this is the strongest practical argument for §6:

1. **Separated relaxation loss from solver failure** — at §9.1's $r=4$, a certified gap of
   **0.002%** beside **22.9%** reconstruction error means the $\ell_1$ solution is genuinely far
   from the $\ell_0$ optimum, not that the optimizer stalled.
2. **Detected its own invalidity** — a *negative* duality gap is impossible for a valid bound,
   flagging that the grid-based supremum was underestimating $\max|\eta|$. A heuristic stopping
   rule fails silently there.
3. **Exposed the norm bias (§7.1.1)** — a certified-converged solve still leaving 61% error is
   a contradiction, forcing attention to the dictionary rather than the optimizer.

### 9.5 Methodological warnings

Three confounds were invisible in design and obvious only in data:

- **Resolution tracked overlap.** Varying width to control separation also changed how well
  atoms were resolved; every method improved together, including the control. Fix: hold width
  fixed in pixels, vary spacing.
- **The normalization inflates.** $\tfrac12\|y\|^2$ grows ~4× as atoms merge, since same-sign
  atoms sum constructively. Report absolute error too.
- **The control can fail.** Random-restart (P0) finds nothing when atoms are small and far
  apart — 100% error where certificate-driven greedy was exact.

---

## 10. The central negative result

**[V] On the certified rows — six of them, $N\in\{8,16\}$ across three targets —
$E_{\rm ref}<E_{\rm BLpol}$ every time.** §10.1 attributes $E_{\rm ref}$ to greedy in all
cases, so greedy beat BLASSO *after* debiasing and full polish.

| target | $N$ | greedy | BLASSO + polish | greedy margin |
|---|---|---|---|---|
| cartoon | 8 | **2.096** | 2.500 | 16% |
| cartoon | 16 | **1.148** | 1.200 | 4% |
| ascent | 8 | **8.301** | 9.585 | 13% |
| ascent | 16 | **5.792** | 6.673 | 13% |
| face | 8 | **5.254** | 5.724 | 8% |
| face | 16 | **3.593** | 4.285 | 16% |

**[A] The $N=32$ rows point the same way but are uninformative about the formulation.** §9.2
excludes them because BLASSO did not converge there (certgap 1.5–6.5%, §9.3); a non-converged
solve losing tests the solver, not (P$\lambda$). For the record they were 0.570/4.154/2.433
against 0.715/5.129/2.653 — margins of 25%, 23%, 9%, consistent with the certified rows.
Reporting them as part of a "nine of nine" result, as an earlier version did, double-counted
a solver defect as evidence about the formulation.

### 10.1 Attribution — resolved

$E_{\rm ref}$ is a minimum over greedy, random restarts and polished BLASSO, and the original
run did not log which attained it. Re-running the two reference methods separately settles it:

| target | $N$ | greedy | restarts | winner |
|---|---|---|---|---|
| cartoon | 8 / 16 / 32 | **2.096 / 1.148 / 0.570** | 2.521 / 1.757 / 1.109 | greedy |
| ascent | 8 / 16 / 32 | **8.301 / 5.792 / 4.154** | 9.602 / 7.805 / 5.252 | greedy |
| face | 8 / 16 / 32 | **5.254 / 3.593 / 2.433** | 7.263 / 4.735 / 3.381 | greedy |

**[V]** Greedy wins all nine, and its value reproduces $E_{\rm ref}$ **exactly** in every row —
so greedy, not restarts, attained the reference throughout. This comparison does not involve
BLASSO, so all nine rows are valid here.

**Established chain**, on the six certified rows: **greedy > polished BLASSO > raw BLASSO**
(polish improves BLASSO in every row, and greedy still beats it). Separately, **greedy >
random restarts** on all nine.

**[A] So the operative ingredient is placement, not relaxation.** Certificate-driven greedy at
$\lambda=0$ is matched-filter placement plus local refinement — matching pursuit. Adding
$\ell_1$ on top degrades quality everywhere tested. §9.1 gives the extreme case: the same
greedy recovered four atoms exactly (0.0000%) where random restarts scored 100%.

**[A] Consequences for this document's framing.**

- (P$\lambda$) is **not** justified as the working definition of optimal encoding. Demoted to a
  diagnostic instrument.
- The actionable distillation is **matching pursuit with continuous refinement over a
  unit-norm anisotropic Gaussian dictionary** — which **[V]** `papers/tip2006.pdf` did in 2006.
  §10.1 establishes the attribution.
- The certificate's value is in *measurement and debugging* (§9.4), not in producing encodings.

**[A] What would overturn this.** The comparison is at $N\le32$ on small images, one instance
per cell, with a BLASSO solver having known defects (§9.3). A better-engineered BLASSO at
realistic $N$ could plausibly close or reverse the gap. §11 U7.

---

## 11. Open questions, and what would settle each

| | question | what would settle it | cost |
|---|---|---|---|
| **U1** | Does GaussianImage's implementation clamp or otherwise break linearity (§3.1)? | Read the released CUDA kernel and loss; check for output clamping or an activation on $c'_n$ | hours |
| **U2** | Does the certificate generalize to colour (§3.3)? | Derive the group-BLASSO dual, verify $\|\eta(\theta)\|_2\le1$; obtain a primary source on vector-valued BLASSO | days |
| **U3** | Is A1's boundarylessness essential or technical (§7.1)? | Check whether CPGD's proof localizes when the optimum's scales are interior; failing that, run CPGD on truncated $\Theta$ and test against the predicted linear rate | weeks |
| **U4** | Is the recovery/approximation mechanism behind the hump real (§9.1)? | Hold $K_{BL}$ exactly at $K$ with a support-constrained solve and re-sweep separation; if the hump persists, $\lambda$-matching and count drift are excluded | days |
| **U5** | Can the solver certify at realistic $N$ (§9.3)? | Replace add/prune with batched addition and a support-aware prune rule, or an exact LASSO inner solve; target certgap $<0.1\%$ at $N=10^3$ | weeks — **gates U7, U8** |
| **U6** | Does §4's rate describe real solutions? | Fit a cartoon target at several $N$; regress $\log$(minor axis) on $\log$(major axis) for edge atoms — parabolic scaling predicts slope 2. Descriptive only; absence is not evidence of failure (§4.1) | days |
| **U7** | Does §10's negative result survive scale and a better solver? | Re-run §10 after U5 at $N\ge10^3$ on $\ge256^2$ images, ≥5 instances per cell with error bars | weeks, after U5 |
| **U8** | Do any §9 conclusions survive $10^3$–$10^5$ atoms? | Re-run §9 after U5 | after U5 |
| **U9** | Is $L^2$-optimal perceptually acceptable (§3.2)? | Compare $L^2$-optimal against SSIM-trained fits at equal $N$, human or perceptual metric | days; needs a metric decision |
| **U10** | Does matching a Zador-type density actually reduce reconstruction error (§7.2)? | Place $N$ atoms by semi-discrete OT at several candidate density laws; compare resulting $L^2$ error against matched-filter placement at equal $N$. If no law wins, the OT guarantee is rigorous about an irrelevant objective | days |
| **U11** | Is the §9.1 hump present out-of-model? | §9.2 sweeps budget, not separation, so the two are not comparable. Sweep $r_{\rm ref}$ on a fixed out-of-model target by varying $N$ and image scale together, and check for a peak | days |
| **U12** | How much of the out-of-model penalty is $\ell_1$ bias vs. no exact representation? | Repeat §9.2 on targets that *are* exact mixtures but at matched $r_{\rm ref}$ and $N$; the difference isolates the model-class effect | days |
| **U14** | How far is *greedy* from the (P0) optimum out-of-model? **Nothing here bounds this.** §10 is a relative result: greedy beats BLASSO, but both could be far from optimal | At small $N$ on small images, a fine exhaustive grid over $(\mu,\Sigma)$ with least-squares amplitudes gives a genuine lower bound on the (P0) optimum; compare greedy against it. Alternatively run greedy with orders-of-magnitude more restarts and refinement and see whether anything improves — weaker, but cheap | days |
| ~~U13~~ | ~~Which uncertified method beat BLASSO — greedy or restarts?~~ | **Resolved — §10.1.** Greedy, in all nine rows, matching $E_{\rm ref}$ exactly | done |

**[A] Dependency structure.** U5 gates U7 and U8, and those two decide whether this line of work
has a future. U1 is cheap and should be first, since §3.1 is a premise for everything. U4, U6,
U10 and U14 are cheap and independent.

**[A] U14 is the one that most limits what can currently be claimed.** §10 establishes only a
*relative* ordering. If greedy is itself far from the (P0) optimum, then "use matching pursuit"
is a statement about the best of two mediocre options, not a recommendation. No result in this
document bounds greedy's absolute suboptimality on a target outside the model class.

---

## 12. Programme

**P1 — U1.** Confirm the forward model. Cheapest check on the load-bearing premise.

**P2 — U5.** Fix the add/prune loop until it certifies at $N\ge10^3$. Nothing at realistic
scale is measurable until this works.

**P3 — U7/U8.** Re-run §10 and §9 at scale with error bars. If greedy still wins, the honest
conclusion is that the convex route is a diagnostic tool and matching pursuit is the encoder.

**P4 — U4, U6.** Cheap, independent, and they test the two interpretive claims currently
resting on hypothesis.

**P5 — geometry-informed initialization.** Structure tensor or $|H|$ → density law →
semi-discrete OT placement → Hessian-derived covariances. **[A]** Motivated by §7.2, not implied
by it — those guarantees are for a surrogate objective. The incumbent to beat is matched-filter
placement, not random initialization.

**Reference points.** **[V]** `papers/tip2006.pdf` — matching pursuit over translated, rotated,
anisotropically-scaled atoms, reported comparable to JPEG2000 and SPIHT at low rates.
**[V\*]** Its dictionary is *richer* than the splatting one (a Gaussian along one axis × second
derivative of a Gaussian in the orthogonal axis, plus a pure-Gaussian sub-dictionary).
**[A]** Reported on a bits axis, not an atom-count axis, so not directly usable as a numerical
baseline. **[V]** `papers/gbc.pdf` gives the one atom-axis datum: ~1024 anisotropic Gaussians
for a recognizable natural image.

---

## Appendix A: methodology and error history

Rules adopted after auditing this document's own reversals. Ten substantive claims were stated
and later withdrawn.

**M1 — Instrument before interpretation.** *Cost of violating:* a solver bug (§7.1.1)
invalidated two complete sweeps and three interpretations; the one-run sanity check that
exposed it was performed last.
**M2 — One observation is a hypothesis.** *Cost:* four reversals, each refuted by the next data
point.
**M3 — Sufficient is not necessary.** Failing a sufficient condition withdraws a guarantee; it
does not make the conclusion false. *Cost:* §8's original conclusion, and a threshold
measurement placed at 4–8 rather than ~30.
**M4 — Tag conversation as well as prose.**
**M5 — Pre-register what an experiment can and cannot show.** *Worked:* E2b's lower-bound
asymmetry never needed revision. *Not done:* E2, corrected three times mid-flight.
**M6 — A null search result is not novelty.** *Cost:* two bodies of prior art missed
(Goal-Based Caustics 2011, `tip2006.pdf`), both having already solved problems treated as open.
**M7 — Analyse in batches.**

**Why this document is exposed to M3.** The imported theory and the target problem differ on
four independent axes — fixed vs free kernel, separated vs dense, recovery vs approximation,
constant vs varying atom norm. Every theorem cited breaks on at least one, so failed hypotheses
are common and easy to misread as negative results about the problem rather than statements
about the reach of a proof. Three of five audit fixes were places where the document had
claimed a *stronger negative* than the theorem licensed; the bias ran pessimistic.

---

## Appendix B: sources

PDFs in `papers/`; citations in `papers/README.md`.

| Tag | File | Used for |
|---|---|---|
| **[V]** | `1910.10319v1.pdf` | dictionary definition, $N$-term rates, cartoon class |
| **[V]** | `1811.06416v1.pdf` | BLASSO, dual certificate, SFW, KER(k), finite termination |
| **[V]** | `1907.10300v2.pdf` | CPGD, WFR flow, Theorem 4.2, (A1–5), $\alpha/\beta$ |
| **[V]** | `1805.09545v2.pdf` | mean-field / OT framework |
| **[V]** | `2509.12889v5.pdf` | BLASSO for GMM, unknown **diagonal** covariances; Thm 5.1 separation |
| **[V]** | `ma.pdf` | semi-discrete OT, damped Newton, global linear convergence |
| **[V]** | `dGBOD12.pdf` | CCVT as OT, power diagrams, concave maximization |
| **[V]** | `Balzer_etal_2009_CCPDAVoLM.pdf` | CCVT definition |
| **[V]** | `MSR-TR-2009-174.pdf` | fast CCVT |
| **[V]** | `BLdG+16.pdf` | Hessian-based anisotropy, Bregman diagrams |
| **[V]** | `gbc.pdf` | CCVT + EM precedent, ~1024 kernels, initialization comparison |
| **[V]** | `tip2006.pdf` | matching pursuit reference, atom definition, unit-norm convention |
| **[V]** | `2403.08551v5.pdf` | Eq. 7 linearity, $L^2$ loss, no density control, initialization |

**Not read — all [U]:** SteepGS; Zador / quantization theory; anisotropic mesh adaptation;
sparse-approximation hardness; group BLASSO; Structure-Guided Allocation; Image-GS.

**Closest theoretical result to the target problem.** **[V]** `2509.12889v5.pdf` extends BLASSO
to Gaussian mixtures with component-specific unknown **diagonal** covariances. Two gaps:
rotation is not covered, and the setting is density estimation from i.i.d. samples rather than
$L^2$ approximation of a signal.

**Literature search note.** "2D Gaussian Splatting" is ambiguous; most results — including
nearly all robotics, satellite and medical work — refer to 2D Gaussian *surfels embedded in 3D*
(Huang et al., SIGGRAPH 2024), a different problem.
