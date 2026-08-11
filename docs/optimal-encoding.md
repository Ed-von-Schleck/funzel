# Optimal encoding of images as 2D Gaussians

Research notes. Goal: given a target image, find the *best possible* representation as a
set of 2D Gaussians — and know how far from best any given representation is.

This is not a compression document. Rate, entropy coding, decode speed and codec
benchmarks are out of scope except where they bear on encoder quality.

Every claim below is tagged with its verification status:

- **[V]** verified against the primary source in `papers/`
- **[V*]** verified, with a caveat stated inline
- **[A]** my analysis — reasoning from verified facts, not itself cited
- **[U]** unverified; from search summaries only, primary source not read

---

## 0. Summary

The single most useful result: **the encoding problem admits a convex reformulation with a
computable optimality certificate.** You can bound the suboptimality of an encoding without
exhaustive search. Fixed-$N$ Gaussian splatting as practised has no such object.

Three independently verified strands converge on one algorithm shape:

| Component | Supplied by | Guarantee |
|---|---|---|
| Objective + certificate | BLASSO | convex; $\|\eta_\lambda\|_\infty \le 1$ certifies optimality |
| Where to add atoms | dual certificate $\eta_\lambda$ | exact Frank-Wolfe step |
| Optimizer | Conic Particle Gradient Descent | global convergence under stated conditions |
| Initialization | semi-discrete OT + Hessian anisotropy | globally convergent damped Newton |
| Split rule | SteepGS | provably minimal offspring count |
| Achievable rate | anisotropic Gaussian approximation theory | curvelet-optimal $N$-term rates |

Nobody has assembled these. That assembly is the project.

---

## 1. Problem statement

Three distinct notions of "optimal encoding". They are not equivalent and only one is
tractable.

**(P0) Best $N$-term approximation.** For a target image $I$ and budget $N$:

$$\min_{\{c_i,\mu_i,\Sigma_i\}_{i=1}^N} \Big\| I - \sum_{i=1}^N c_i\, G(\mu_i,\Sigma_i) \Big\|^2$$

The natural formulation, and the one "optimal encoding" intuitively means. Intractable —
see §3.

**(P$\lambda$) Rate-regularized encoding over measures.** Let $\Theta = \mathbb{R}^2 \times \mathrm{SPD}(2)$
be the atom parameter space, $\varphi_\theta$ the corresponding Gaussian, and $m$ a Radon
measure on $\Theta$:

$$\min_{m \in \mathcal{M}(\Theta)} \tfrac12 \|\Phi m - I\|^2 + \lambda |m|(\Theta), \qquad \Phi m = \int_\Theta \varphi_\theta \, dm(\theta)$$

**[V]** — this is the BLASSO, exactly as stated in `papers/1811.06416v1.pdf`. $|m|(\Theta)$
is the total variation norm, the continuum analogue of $\ell_1$: for $m = \sum_i c_i \delta_{\theta_i}$
it equals $\sum_i |c_i|$.

**Convex.** Infinite-dimensional, but convex. No spurious local minima. No permutation
symmetry — a measure has no ordering. $N$ is not a hyperparameter; it emerges from $\lambda$.

**(P∞) Asymptotic achievable rate.** How fast can error decay with $N$ for a given image
class? Answered in §2.

**Relationship.** (P$\lambda$) is a convex relaxation of (P0), the continuum version of the
$\ell_1$/$\ell_0$ relaxation. Under separation and non-degeneracy conditions it recovers the
correct support exactly, but in general there is a relaxation gap. **[A]** Adopting (P$\lambda$)
as the working definition of "optimal encoding" is a real choice, and its justification is
that it is the only one of the three you can certify.

---

## 2. What the optimum looks like

**[V]** `papers/1910.10319v1.pdf` — Erb, Hangelbroek & Ron, *Anisotropic Gaussian
approximation in $L_2(\mathbb{R}^2)$*.

The dictionary studied is exactly the splatting dictionary:

$$\mathcal{D} := \{\varphi \circ M \mid M \in GL(d,\mathbb{R}) \ltimes \mathbb{R}^d\}, \qquad x \mapsto e^{-|A(x-k)|^2}$$

with $A$ any invertible matrix — full anisotropy including rotation, not diagonal.

Smoothness class $\mathcal{C}^\alpha$ = images whose decreasing-rearranged curvelet
coefficients satisfy $|\omega_n| = O(n^{-\alpha})$.

**Theorem (Thm 1/2).** For $\alpha > 1/2$ and $f \in \mathcal{C}^\alpha$, there is an $N$-term
Gaussian mixture $T_N f = \sum_{j=1}^N a_j G_j$, $G_j \in \mathcal{D}$, with

$$\|T_N f - f\|_2 \le C_f\, N^{\frac12 - \alpha}$$

Curvelet $N$-term thresholding gives the **same** bound $C N^{\frac12-\alpha}$. For the
Donoho–Candès cartoon class ($f_1 + f_2\chi_\Omega$, both $C^2$, $\partial\Omega$ a $C^2$
curve), curvelets give $O(N^{-1}(\log N)^{3/2})$ and Gaussian mixtures give
$O(N^{-1}(\log N)^{3/2})$ — identical, no log penalty. Curvelet rates are optimal
essentially by definition for this class.

Lemma 3: approximating a single curvelet with $M$ Gaussians has error $C_K M^{-K}$ for
**every** $K$ — super-algebraic, so the per-curvelet constant is benign.

**[V*] Important caveat.** The construction routes through the curvelet transform, and
$m^*$ and the sub-budgets $N(n)$ are *independent of $f$* — only the coefficients $\omega_n$
are data-dependent. The Gaussian geometry is fixed by the curvelet tiling, not adapted to
the image. This is an **achievability** result for the dictionary, not a competing encoder.
An adaptive encoder searches a superset, so the bound stands as a floor on what is possible.

**[A] What this implies for the project.** The primitive is not the bottleneck. Anisotropic
2D Gaussians achieve provably optimal rates for edge-dominated images. Any shortfall in a
real system is an *encoder* failure. It also predicts what a good solution looks like:
multiscale, edge-aligned, strongly anisotropic, with parabolic scaling (curvelet-like
tiling) — a concrete, testable structural prediction to check against what an optimizer
actually produces.

---

## 3. Why (P0) is hard

- **[U]** Sparse approximation over a general dictionary is NP-hard, and NP-hard to
  approximate within any factor; specifically hard under **coherent** dictionaries. The
  Gaussian dictionary is highly coherent — nearby atoms are nearly parallel.
- The dictionary is **continuous**: positions, scales and rotation are real-valued. Not
  "choose $N$ of $K$" but an infinite-dimensional non-convex search.
- **[A] Permutation symmetry**: (P0) has $N!$ equivalent global minima. "The" optimum is an
  equivalence class, and the landscape carries symmetry-induced saddles.
- **[U]** SteepGS finds empirically that stuck primitives sit at **saddle points**, not local
  minima — gradients vanish but loss is still reducible, just not along any direction a
  single primitive can move.
- **[V]** Goal-Based Caustics states the practitioner's version plainly: *"while EM is
  guaranteed to converge, it will generally only find a locally optimal solution. Hence it
  is important to supply it with a reasonable initial parameter estimate."*

---

## 4. The structural fact that makes the theory apply

**[A] This is the load-bearing observation of the whole investigation.**

GaussianImage's rasterizer replaces depth sorting and $\alpha$-blending with **accumulated
summation** — each pixel is the plain weighted sum of 2D Gaussians. Therefore

$$I(p) = \sum_i c_i\, G_{\theta_i}(p)$$

is **linear in the coefficients** $c_i$, with atoms indexed by $\theta = (\mu,\Sigma)$. That
is precisely $\Phi m$ for $m = \sum_i c_i \delta_{\theta_i}$. **2D Gaussian splatting with
accumulated summation is a BLASSO forward model.**

3D Gaussian splatting is not: $\alpha$-blending makes the render nonlinear in the
primitives (occlusion), so the measure-space convexification does not apply.

**[A]** GaussianImage dropped $\alpha$-blending for decode speed and, apparently without
noticing, made the problem convexifiable. Everything in this document applies to 2D
specifically *because* of that choice, and would not transfer to 3D.

Two adaptations needed:
- RGB is vector-valued → group BLASSO. **[U]** Exists in the literature.
- Signed colors are fine (TV handles signed measures). Non-negative colors would put us in
  the **stronger** positive-measure theory. **[V]** SFW has an explicit positive variant: the
  stopping test becomes $\eta \le 1$ and the LASSO is solved on the positive orthant.

---

## 5. The optimality certificate

**[V]** From `papers/1811.06416v1.pdf`. Define

$$p_\lambda = \tfrac1\lambda (I - \Phi m_\lambda), \qquad \eta_\lambda = \Phi^* p_\lambda$$

Optimality is characterized by $\|\eta_\lambda\|_{\infty,\Theta} \le 1$, with
$\eta_\lambda(\theta_i) = \mathrm{sign}(c_i)$ at support points.

**What $\eta_\lambda$ is:** the residual correlated against **every candidate atom in the
family**. It is a function over parameter space $(\mu,\Sigma)$, not over pixels. Its
argmax says *place a Gaussian here, with this covariance*.

**Two things this buys, both central to the project goal:**

1. **A stopping test.** $\|\eta_\lambda\|_\infty \le 1$ certifies global optimality of the
   (P$\lambda$) solution. Computable.
2. **A suboptimality bound.** **[V]** SFW inherits the Frank-Wolfe rate
   $T_\lambda(m^{[k]}) - T_\lambda(m^\star) \le C_1/k$, and FW duality gaps bound the
   distance to optimum at every iteration. **[A]** So you can state "this encoding is within
   $\varepsilon$ of optimal" without ever finding the optimum — far stronger than estimating
   a lower envelope by random restarts.

**[A] Contrast with current practice.** Densification heuristics in 2D GS — Image-GS samples
proportional to *pixel-space* error, GaussianImage++ uses distortion-driven growth, 3DGS
uses positional-gradient magnitude — are all surrogates for $\eta_\lambda$, and none of them
see the covariance dimension. They choose *where*, then let SGD find the shape. The dual
certificate chooses where **and what shape** in one argmax, and it is the provably correct
object.

**[V*] Representer theorem.** TV regularization yields solutions supported on at most
(number of measurements) atoms. For images, measurements = pixels, so the bound is
$N \le \#\mathrm{pixels}$ — vacuous. It becomes meaningful only under sketching or
band-limiting of the measurements. Do not expect sparsity for free from this.

---

## 6. Algorithms with guarantees

### 6.1 Sliding Frank-Wolfe

**[V]** `papers/1811.06416v1.pdf`, Algorithm 2:

| Step | Operation | Nature |
|---|---|---|
| 3 | $\theta_* = \arg\max_\theta \lvert\eta^{[k]}(\theta)\rvert$ | add one atom |
| 7 | LASSO on amplitudes, positions frozen | convex (FISTA) |
| 8 | joint solve over amplitudes **and** positions | non-convex (bounded BFGS) |
| 9 | *"remove zero amplitudes Dirac masses"* | prune |

Stopping: $|\eta^{[k]}(\theta_*)| \le 1$. Step 3 is implemented as grid search + Newton.

**Theorem 3 (finite termination).** If $m_{a,x}$ is the *unique* solution of (P$\lambda$) and
$\eta_\lambda$ is nondegenerate —

$$\forall \theta \notin \mathrm{supp}, \ |\eta_\lambda(\theta)| < 1 \qquad\text{and}\qquad \forall i, \ \eta_\lambda''(\theta_i) \neq 0$$

— then the algorithm recovers $m_{a,x}$ in a **finite** number of steps.

The paper is explicit that joint (not alternating) optimization in step 8 is what buys
finite termination.

**[V*] Caveat:** *"we state and prove this Theorem in the case of $d=1$ but the changes for
$d \in \mathbb{N}^*$ can be easily done."* Asserted, not shown, for our setting.

**[A] Why SFW is not the algorithm.** One atom per iteration, each followed by a full BFGS
re-solve over all parameters. At $10^4$–$10^5$ atoms that is hopeless. **The transferable
idea is the dual certificate as densification criterion**, batched — add many atoms per
round at local maxima of $|\eta|$ — not the iteration schedule.

**[A] Note the structural identity anyway:** add / fit / slide / prune *is* densify → optimize
→ prune. Current GS reinvented the Frank-Wolfe schedule without the certificate that says
where, or the theorem that says when to stop.

### 6.2 Conic Particle Gradient Descent — the answer to "are we stuck in local optima?"

**[V]** `papers/1907.10300v2.pdf` — Chizat. Problem form:

$$J(\nu) = R\Big(\int_\Theta \varphi(\theta)\,d\nu(\theta)\Big) + \lambda\,\nu(\Theta)$$

Method: **discretize the measure into particles and run non-convex gradient descent on
positions and weights.** That is exactly what splatting does.

Update, with $\alpha$ the **weight** step-size and $\beta$ the **position** step-size:

$$(r_i,\theta_i) \leftarrow \mathrm{Ret}_{(r_i,\theta_i)}\big(-2\alpha r_i J'_\nu(\theta_i),\ -\beta \nabla J'_\nu(\theta_i)\big)$$

For $\alpha,\beta>0$ this is the gradient flow of $J$ in the **Wasserstein–Fisher–Rao**
metric. Under the mirror retraction the weight update is $r\exp(\delta r/r)$ —
**multiplicative**, i.e. exponentiated gradient — while positions update additively.

**Theorem 4.2 (global convergence of gradient descent).** Under (A1–5), with $\rho$ an
absolutely continuous reference measure with smooth positive density and $\log\rho$
Lipschitz, if

$$W_\infty(\nu_0,\rho) \le (J_0-J^\star)/C, \qquad \alpha \le (J_0-J^\star)^{1+\epsilon/2}/C, \qquad \beta \le (J_0-J^\star)\alpha^2/C'$$

then projected gradient descent from $\nu_0$ converges to the **global** optimum $\nu^\star$,
linearly after $k_0$.

Complexity scales as $\log(1/\epsilon)$ rather than $\epsilon^{-d}$ for grid-based convex
methods.

**[V*] The cost.** $\nu_0$ is an $m$-sample empirical approximation of $\rho$, and
$W_\infty$ convergence is $\tilde O(m^{-1/d})$. The paper states this exponential dependence
on dimension is **unavoidable**. Also $m > m^\star$ is required — genuine overparameterization
— and $J_0 - J^\star$ shrinks as problems get harder, forcing $m$ up.

**[A] Why this is survivable here.** $d = \dim\Theta = 5$ for 2D Gaussians (2 position + 2
scale + 1 rotation). For 3DGS $d$ is far larger. Together with the $\alpha$-blending
nonlinearity, this is the second independent reason the theory fits 2D and not 3D.

### 6.3 Semi-discrete optimal transport — the placement subproblem

**[V]** `papers/ma.pdf` — Kitagawa, Mérigot & Thibert. A damped Newton algorithm for
semi-discrete OT (absolutely continuous source, finitely supported target) with **global
linear convergence**, under (a) the Ma–Trudinger–Wang condition on the cost, and (b)
quantitatively connected support of the source density (a weighted Poincaré–Wirtinger
inequality). Quadratic cost is covered; some non-convex supports also qualify.

**[V]** `papers/dGBOD12.pdf` — de Goes et al. recast capacity-constrained Voronoi
tessellation as semi-discrete OT. The optimal partition is a **power diagram** (Laguerre
cells), not a plain Voronoi diagram, and the paper enforces capacity constraints exactly via
*"a concave maximization w.r.t. the weights via a step-adaptive Newton method"*, with cost
of order a single Newton step. Handles arbitrary density $\rho$.

**[V]** `papers/Balzer_etal_2009_CCPDAVoLM.pdf` — original CCVT: capacity of a point is the
area of its Voronoi region weighted by $\rho$; the constraint is that all capacities are
equal. **[V]** `papers/MSR-TR-2009-174.pdf` — Fast CCVT, orders of magnitude faster,
scales well in point count.

**[A]** So "place $N$ atoms to match a prescribed density" is a *solved* problem with a
globally convergent algorithm. This is strictly stronger footing than any densification
heuristic in the splatting literature.

### 6.4 Covariance from the Hessian

**[V]** `papers/BLdG+16.pdf` — Budninskiy et al., *Optimal Voronoi Tessellations with
Hessian-based Anisotropy*: a variational method generating cell complexes with local
anisotropy conforming to the Hessian of a given function, **for any given local mesh
density**. Built on approximation theory, an anisotropic extension of CVT, dual to Optimal
Delaunay Triangulation. Uses **first-type Bregman diagrams** — power diagrams whose sites
carry a scalar weight *and* a vector-valued shift.

**[V*] Caveat:** the method is stated for the Hessian of a **convex** function. Image
Hessians are indefinite. **[A]** Standard fix from anisotropic mesh adaptation: use $|H|$
(eigenvalue absolute values) or the structure tensor, both PSD.

**[U]** Anisotropic mesh adaptation theory: for minimizing interpolation error the optimal
metric is defined by the (modified) Hessian; the tessellation is locally aligned with the
**eigenvectors** of the Hessian and the anisotropic quotients equal those of the Hessian.

**[A]** This is the principled answer to *what covariance should each Gaussian have*:
orientation from Hessian eigenvectors, aspect ratio from the eigenvalue ratio, scale from
cell size. Note that Structure-Guided Allocation's regularizer aligns Gaussians to local
**gradient** directions — a first-order approximation of a rule the theory says is
second-order. **[U]** That paper already reported large gains with the weaker version.

**[U] Density law.** Zador's theorem: the empirical distribution of the $N$-point optimal
quantizer converges to density $\propto \varphi^{d/(d+s)}$; for $d=2,s=2$ that is
$\propto\sqrt{\varphi}$. **[A] Caveat:** Zador is nearest-neighbour *quantization*
(piecewise-constant), not Gaussian-kernel approximation. Treat as a design principle to
validate empirically, not a theorem about splatting.

### 6.5 EM, and the one empirical precedent

**[V]** `papers/gbc.pdf` — Papas et al., *Goal-based Caustics* (2011). Non-negative image
decomposition into overlapping **anisotropic Gaussian kernels**:

1. sample the image intensity densely ($n = 4{,}000{,}000$ points)
2. compute a **CCVT**; take each Voronoi centroid as a Gaussian center
3. initialize isotropic $\Sigma_i = \mathrm{diag}(\sigma^2,\sigma^2)$, $\sigma$ = radius of a
   $k$-NN query with $k$ = the region's capacity
4. refine with a **sparse EM**

Reported at ~**1024 Gaussians** per image. They always use anisotropic Gaussians, citing
"significant qualitative improvement" over isotropic.

**[V] The relevant sentence:** they experimented with several initialization strategies and
*"obtained the most reliable results using regular point sets that adapt to the image
intensity"* — i.e. CCVT won an empirical initialization bake-off, in 2011, for exactly this
problem. This is the closest existing precedent to the proposed program.

### 6.6 Saddle-escape splitting

**[U]** SteepGS derives, from optimization theory: necessary conditions for densification to
reduce loss; the **minimal number of offspring is exactly 2**; the optimal displacement
direction is the minimal eigenvector of a per-primitive *splitting matrix*, triggered when
its eigenvalue is negative; and an analytic offspring-weight normalization. Reported ~50%
fewer primitives at equal quality. Derivation is primitive-local — **[A]** should port to 2D
almost mechanically. Primary source not read.

---

## 7. Design rules the theory implies

**[A] throughout, derived from the verified results above.**

**R1 — Initialization must have full support.** CPGD's global convergence requires $\nu_0$
to be $W_\infty$-close to a reference $\rho$ with *smooth positive density* over all of
$\Theta$, densely sampled, with $m > m^\star$.

This **inverts** the usual initialization intuition. GaussianImage's "unsophisticated"
uniform-random initialization over positions *and* covariances is precisely what the theorem
asks for. Image-GS's gradient/saliency-concentrated initialization **violates** the
full-support condition — it buys convergence speed by giving up the guarantee.

But $\rho$ is explicitly the *prior on the solution*, and smaller $\bar H(\nu^\star,\rho)$
improves the bounds. **The correct design is a Hessian/density-law-informed $\rho$ that
remains strictly positive everywhere** — neither uniform nor concentrated. Neither existing
encoder does this.

**R2 — Weight updates should be multiplicative.** The WFR/Fisher-Rao component acts
multiplicatively on mass ($r\exp(\delta r/r)$). Current encoders use additive Adam on
colors. Testable divergence.

**R3 — Positions must move slower than weights.** Global convergence needs $\beta/\alpha$
small. Current GS practice already uses position LR $\ll$ color LR, so this is accidental
compliance; it should be made deliberate and tuned against the condition.

**R4 — Densify at $\arg\max_\theta|\eta(\theta)|$, not at pixel-error maxima.** And search
over covariance as well as position.

**R5 — Densification is not a hack.** WFR flow has a transport term (moves mass) and a
Fisher-Rao term (creates/destroys it). Adam on positions discretizes the first; clone/split
and prune discretize the second. Densification is the half of the correct flow that plain
SGD cannot express.

**R6 — Split into exactly 2, along the minimal eigenvector**, when the splitting matrix has
a negative eigenvalue (§6.6).

**R7 — Consider a quasi-Newton slide.** SFW uses bounded BFGS for the joint
amplitude+position step, not SGD. **[U]** Second-order splatting optimizers report large
gains when primitive count is low — which is the 2D regime.

---

## 8. Gaps

Ordered by how much they threaten the program.

1. **SFW finite termination is proved in $d=1$ only.** Extension asserted, not shown.
2. **BLASSO for Gaussians with unknown covariance covers diagonal only.** **[V]**
   `papers/2509.12889v5.pdf` extends BLASSO to GMMs with component-specific unknown
   **diagonal** covariances, with non-asymptotic recovery for means, covariances and weights
   via a Fisher-Rao semi-distance and an explicit separation condition. **Rotation is not
   covered**, and splatting uses rotation + two scales. Narrow, real, and the obvious
   theoretical target.
3. **That result is density estimation from i.i.d. samples, not $L^2$ signal approximation.**
   Different noise model. A single-channel non-negative image is closer than it looks, but
   it is not the same theorem.
4. **Exponential-in-$d$ particle count** for CPGD's global guarantee. $d=5$ makes it
   plausible but not free.
5. **Zador is not a splatting theorem.**
6. **$\ell_1$/$\ell_0$ relaxation gap** between (P$\lambda$) and (P0) is uncharacterized here.
7. **No published measurement of how suboptimal existing encoders are.**

---

## 9. Program

**E1 — Measure the gap.** The founding experiment. On small images and small $N$, compute
(a) the BLASSO optimum with certificate, and (b) what GaussianImage / Image-GS / EM actually
achieve. Report the certified distance to optimal. Nobody has published this, and §5 makes
it computable rather than merely estimable.

**E2 — Certificate-driven densification.** Replace error-magnitude sampling with batched
$\arg\max_\theta|\eta(\theta)|$ over a coarse covariance grid. Directly testable against
existing densifiers at equal $N$.

**E3 — OT initialization.** Structure tensor / $|H|$ field → density law → semi-discrete OT
placement (globally convergent damped Newton) → Hessian-derived covariances → strictly
positive $\rho$ per R1. Drop in as an additional init mode alongside `gradient`, `saliency`,
`random` and compare at equal $N$ and equal iteration budget.

**E4 — CPGD-conformant optimizer.** Multiplicative weight updates, tuned $\beta/\alpha$,
overparameterized start, per R1–R3.

**E5 — Port SteepGS's split rule to 2D.**

**E6 — Check the structural prediction.** Does a well-optimized solution actually exhibit
curvelet-like tiling — multiscale, edge-aligned, parabolic scaling (§2)? If not, either the
optimizer is failing or the image class assumption is wrong. Cheap diagnostic, informative
either way.

**Baselines with published numbers.** **[V]** `papers/tip2006.pdf` — Figueras i Ventura,
Vandergheynst & Frossard (2006) — is the only principled-encoder baseline found. Matching
pursuit over a dictionary of translated, rotated, anisotropically-scaled atoms. **[V*] Note
the dictionary is *richer* than the splatting one**: the primary atom is a Gaussian along one
axis × **second derivative of a Gaussian** in the orthogonal axis, with a separate pure-Gaussian
sub-dictionary for low frequencies. Erb–Hangelbroek–Ron explicitly place such modulated atoms
*outside* dictionary $\mathcal{D}$.

Its reported results (PSNR, dB):

| Image | Rate | MP | JPEG2000 | SPIHT |
|---|---|---|---|---|
| Lena 256² | 0.35 bpp | 30.36 | **30.79** | **31.35** |
| Lena 512² | 0.16 bpp | 31.06 | **31.93** | — |
| Goldhill 256² | 0.23 bpp | 27.49 | **28.18** | — |
| Barbara 256² | 0.12 bpp | **21.35** | 21.23 | 21.29 |

**[A]** Reported on a bits axis rather than an atom-count axis, so it does not transfer
directly to (P0). It is included because a *greedy, principled* encoder over a *richer*
dictionary is a meaningful reference point for encoder quality, and because it is the only
one that exists.

---

## Appendix: sources

All PDFs in `papers/`; see `papers/README.md` for full citations.

| Tag | File | Used for |
|---|---|---|
| **[V]** | `1910.10319v1.pdf` | dictionary definition, $N$-term rates, curvelet optimality |
| **[V]** | `1811.06416v1.pdf` | BLASSO, dual certificate, SFW algorithm, finite termination |
| **[V]** | `1907.10300v2.pdf` | CPGD, WFR flow, global convergence conditions, $\alpha/\beta$ |
| **[V]** | `1805.09545v2.pdf` | mean-field / OT framework underlying the above |
| **[V]** | `2509.12889v5.pdf` | BLASSO for GMM with unknown diagonal covariances |
| **[V]** | `ma.pdf` | semi-discrete OT, damped Newton, global linear convergence |
| **[V]** | `dGBOD12.pdf` | CCVT as OT, power diagrams, concave maximization + Newton |
| **[V]** | `Balzer_etal_2009_CCPDAVoLM.pdf` | CCVT definition |
| **[V]** | `MSR-TR-2009-174.pdf` | fast CCVT, scalability |
| **[V]** | `BLdG+16.pdf` | Hessian-based anisotropy, Bregman diagrams |
| **[V]** | `gbc.pdf` | CCVT + sparse EM precedent, ~1024 kernels, init bake-off |
| **[V]** | `tip2006.pdf` | matching pursuit baseline, atom definition, PSNR table |

**[U]** items rest on search summaries only: SteepGS, Zador's theorem, anisotropic mesh
adaptation, sparse-approximation hardness, group BLASSO, Structure-Guided Allocation.
Obtain primaries before relying on them.

### Note on literature search

"2D Gaussian Splatting" is ambiguous. Most results — including nearly all robotics,
satellite and medical work — refer to 2D Gaussian *surfels embedded in 3D* (Huang et al.,
SIGGRAPH 2024), which is a different problem. Filter accordingly.
