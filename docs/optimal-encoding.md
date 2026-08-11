# Optimal encoding of images as 2D Gaussians

Research notes. Goal: given a target image, find the best possible representation as a set
of 2D Gaussians, and know how far from best any given representation is.

Not a compression document. Rate, entropy coding and decode speed are out of scope.

## Status tags

- **[V]** verified against a primary source in `papers/`
- **[V\*]** verified, with a caveat stated inline
- **[U]** unverified — from search summaries only, primary source not read
- **[A]** my analysis, reasoning from **[V]** facts
- **[A←U]** my analysis, resting on at least one **[U]** premise

`[A←U]` claims are the fragile ones. The main one remaining is colour (§2.3).

The load-bearing conclusion is §7: the BLASSO recovery theory is built for sparse,
well-separated atoms, and image encoding is a dense, overlapping regime it does not cover.
The optimality certificate (§5) survives that; almost nothing else does.

---

## 1. Problem statement

Three notions of "optimal encoding". They are not equivalent.

**(P0) Best $N$-term approximation.**

$$\min_{\{c_i,\mu_i,\Sigma_i\}_{i=1}^N} \Big\| I - \sum_{i=1}^N c_i\, G(\mu_i,\Sigma_i) \Big\|^2$$

The formulation "optimal encoding" intuitively means. Intractable (§4).

**(P$\lambda$) Rate-regularized encoding over measures.** With $\Theta$ the atom parameter
space, $\varphi_\theta$ the corresponding Gaussian, $m$ a Radon measure on $\Theta$:

$$\min_{m \in \mathcal{M}(\Theta)} \tfrac12 \|\Phi m - I\|^2 + \lambda |m|(\Theta), \qquad \Phi m = \int_\Theta \varphi_\theta \, dm(\theta)$$

**[V]** This is the BLASSO exactly as stated in `papers/1811.06416v1.pdf`. $|m|(\Theta)$ is
the total variation norm; for $m = \sum_i c_i\delta_{\theta_i}$ it equals $\sum_i|c_i|$.

Convex — infinite-dimensional but convex. No spurious local minima, no permutation
symmetry. $N$ is not a parameter; it emerges from $\lambda$.

**(P∞) Achievable rate.** How fast can error decay with $N$ for a given image class? (§3)

### 1.1 The relaxation gap

(P$\lambda$) is the continuum analogue of the $\ell_1$ relaxation of $\ell_0$. Under
separation and non-degeneracy conditions BLASSO recovers the correct support exactly
**[V]**, but in general (P$\lambda$) and (P0) have different solutions, and **the size of
that gap for the Gaussian dictionary on natural images is unknown**.

This matters more than it might appear. Everything tractable in this document is about
(P$\lambda$). If the gap is large, certifying optimality for (P$\lambda$) says little about
(P0). Measuring the gap is §11 E2, and it should be done early — it is a precondition for
the rest of the program being worth pursuing, not a detail.

**Choosing (P$\lambda$) as the working definition** is defensible on the grounds that it is
the only one of the three that admits a certificate (§5). It is not defensible on the
grounds that it is equivalent to (P0), because that is unestablished.

---

## 2. Preconditions: does the theory apply at all?

Three structural requirements. The first is probably satisfiable, the second is a real
constraint, the third is unresolved.

### 2.1 Linearity of the forward model — satisfied

The measure formulation requires $\Phi$ linear in the atom coefficients.

**[V]** `papers/2403.08551v5.pdf` §3.2. GaussianImage replaces 3D GS's sorted
$\alpha$-blending with accumulated summation. Starting from
$C_i = \sum_n c_n \alpha_n T_n$ with $\alpha_n = o_n\exp(-\sigma_n)$ and
$\sigma_n = \tfrac12 d_n^T\Sigma^{-1}d_n$, they fold $T_n$ into $o_n$, then note that colour
$c_n$ and opacity $o_n$ are both learnable and merge them into a single coefficient:

$$C_i = \sum_{n\in N} c'_n \exp(-\sigma_n) \tag{Eq. 7}$$

No sorting, no accumulated transparency, no separate opacity, no normalization by
accumulated weight, no clamping in the render path.

**[A]** This is exactly $\Phi m$ with $m = \sum_n c'_n \delta_{\theta_n}$,
$\theta = (\mu,\Sigma)$, linear in $c'_n$. The precondition holds.

**[A]** The atom family is also *identical* to the one in §3: GaussianImage uses
$\exp(-\tfrac12 d^T\Sigma^{-1}d)$, Erb–Hangelbroek–Ron use $e^{-|A(x-k)|^2}$, and
$A^TA = \tfrac12\Sigma^{-1}$ ranges over all SPD matrices as $A$ ranges over $GL(2,\mathbb{R})$.
Same set of functions.

**[A]** 3D Gaussian splatting does not satisfy this: $\alpha$-blending is nonlinear in the
primitives because of occlusion. None of this transfers to 3D.

### 2.2 The loss must be Hilbertian — satisfied for the main baseline

The certificate is $\eta_\lambda = \Phi^*p_\lambda$; it requires an adjoint, hence an
inner-product structure. $L^2$ works. **SSIM and L1 do not.**

**[V]** GaussianImage's representation objective is $L^2$: *"we employ the L2 loss function
to optimize the Gaussian parameters."* So the requirement is met by the primary baseline
rather than being a constraint imposed on it.

**[A]** It is still a real restriction on the family of methods this can cover. **[U]**
Image-GS optimizes a perceptual objective (its repo depends on `fused-ssim`), and methods
with SSIM or L1 terms admit no certificate. Any comparison across that boundary requires
re-running under $L^2$.

Whether $L^2$-optimal encodings are perceptually acceptable at low atom counts is a separate
empirical question this framework cannot answer.

### 2.3 Colour — unresolved

The theory is for scalar-valued atoms. The actual problem is three channels sharing one
geometry.

Two formulations:

1. Three independent BLASSO problems. Loses shared geometry, roughly triples atom count.
   Almost certainly wrong.
2. **Group BLASSO**: amplitudes are 3-vectors, penalty $\sum_i \|c_i\|_2$. Preserves shared
   geometry. **[U]** Group/vector-valued BLASSO exists in the literature; not verified that
   it covers this case.

**[A←U]** Under (2) the certificate should generalize with $\eta$ vector-valued and the
optimality condition becoming $\|\eta(\theta)\|_2 \le 1$ — the dual of the group norm. Not
verified, and the separation/non-degeneracy conditions for the vector-valued case are not
known to me.

Until resolved, all guarantees below are for the single-channel case.

---

## 3. What the dictionary can achieve

**[V]** `papers/1910.10319v1.pdf` — Erb, Hangelbroek & Ron.

The dictionary studied is exactly the splatting dictionary:

$$\mathcal{D} := \{\varphi \circ M \mid M \in GL(d,\mathbb{R}) \ltimes \mathbb{R}^d\}, \qquad x \mapsto e^{-|A(x-k)|^2}$$

$A$ any invertible matrix — full anisotropy including rotation.

**Theorem 1/2.** For $\alpha > 1/2$ and $f \in \mathcal{C}^\alpha$, there exists an $N$-term
Gaussian mixture with $\|T_Nf - f\|_2 \le C_f N^{\frac12-\alpha}$. Curvelet $N$-term
thresholding gives the same bound. Lemma 3: approximating a single curvelet with $M$
Gaussians has error $C_K M^{-K}$ for every $K$.

### 3.1 How much this actually says

**Two results of different strength, which should not be conflated.**

**The $\mathcal{C}^\alpha$ result is partly circular.** $\mathcal{C}^\alpha$ is *defined* by
decay of curvelet coefficients ($|\omega_n| = O(n^{-\alpha})$). So it says Gaussians match
curvelets on the class defined by curvelets working well. Informative about the dictionary's
richness, but not evidence that real images live in $\mathcal{C}^\alpha$ for useful $\alpha$.

**The cartoon-class result is not circular.** **[V]** The Donoho–Candès cartoon class
($f_1 + f_2\chi_\Omega$, both $C^2$, $\Omega$ compact with $C^2$ boundary) is defined
*geometrically*, and its curvelet decay $|\omega_n| = O(n^{-3/2}|\log n|^{3/2})$ is a cited
theorem. Curvelets achieve $O(N^{-1}(\log N)^{3/2})$; Gaussian mixtures achieve the same.
This is a genuine statement about an independently-defined image class.

**Three limits on what follows.**

1. **Asymptotic, with unquantified $C_f$.** At $N=10^3$–$10^5$ — the operating regime —
   asymptotic rates with unknown constants constrain very little.
2. **[V\*] Non-adaptive construction.** $m^*$ and the sub-budgets $N(n)$ are *independent of
   $f$*; the Gaussian geometry is fixed by the curvelet tiling, only coefficients are
   data-dependent. This is an achievability result for the dictionary, not an encoder.
3. **[A]** It does **not** support "any shortfall is an encoder failure." An existence result
   with an unbounded constant cannot attribute a finite-$N$ gap to the optimizer. The
   defensible statement is: *the dictionary is not asymptotically limiting for cartoon-like
   images.*

### 3.2 A structural prediction

**[A]** If the theory describes what good solutions look like, an optimizer that reaches
near-optimality should produce curvelet-like structure: multiscale, edge-aligned, strongly
anisotropic, with **parabolic scaling** — width $\propto$ length$^2$. Operationalized in §11 E5.

---

## 4. Why (P0) is hard

- **[U]** Sparse approximation over a general dictionary is NP-hard, and NP-hard to
  approximate within any factor; hardness holds specifically under *coherent* dictionaries.
  The Gaussian dictionary is highly coherent.
- The dictionary is continuous — an infinite-dimensional non-convex search, not a subset
  selection.
- **[A]** (P0) has $N!$ equivalent global minima; the optimum is an equivalence class and the
  landscape carries symmetry-induced saddles.
- **[U]** SteepGS reports that stuck primitives sit at *saddle points*, not local minima —
  loss is still reducible, but not along any direction a single primitive can move.

---

## 5. The optimality certificate

The strongest result available, and the main reason to prefer (P$\lambda$).

**[V]** `papers/1811.06416v1.pdf`:

$$p_\lambda = \tfrac1\lambda(I - \Phi m_\lambda), \qquad \eta_\lambda = \Phi^* p_\lambda$$

Optimality is characterized by $\|\eta_\lambda\|_{\infty,\Theta} \le 1$, with
$\eta_\lambda(\theta_i) = \mathrm{sign}(c_i)$ on the support.

$\eta_\lambda$ is the residual correlated against **every candidate atom in the family** — a
function over parameter space $(\mu,\Sigma)$, not over pixels. Its argmax specifies both a
position and a covariance.

**What this provides:**

1. **A global optimality test for (P$\lambda$)**, computable.
2. **A suboptimality bound at every iteration.** **[V]** SFW inherits the Frank-Wolfe rate
   $T_\lambda(m^{[k]}) - T_\lambda(m^\star) \le C_1/k$; FW duality gaps bound distance to
   optimum without finding it.

**[A] Why these two survive where the rest of the theory does not (§7).** Both are
consequences of convex duality for a TV-regularized problem with a bounded linear $\Phi$.
They need no condition on the kernel, no separation between atoms, and no non-degeneracy.
$\|\varphi_\theta\|$ is uniformly bounded for Gaussians, so $\Phi$ is bounded and the
duality argument applies directly.

This matters because §7 shows that almost everything else in the BLASSO literature is
conditioned on a regime that image encoding is not in. The certificate is the part that
transfers.

**[A] Scope.** Both are statements about (P$\lambda$) under $L^2$ loss for scalar atoms.
Neither certifies (P0) (§1.1), and neither survives a change of loss (§2.2).

**[A] Relation to current practice.** Densification heuristics — pixel-space error sampling,
distortion-driven growth, positional-gradient magnitude — are surrogates for $\eta_\lambda$,
and none of them search the covariance dimension. They choose *where* and let SGD find the
shape. The certificate chooses both in one argmax.

**[V]** GaussianImage has no densification at all: it *"discard[s] adaptive density
control"*, reasoning that *"there is no so-called empty area in the 2D image space"*, and
fixes the atom count in advance. So in the baseline case the comparison is not
certificate-versus-heuristic but certificate-versus-nothing.

---

## 6. Algorithms, and what each guarantee actually covers

### 6.1 Sliding Frank-Wolfe

**[V]** `papers/1811.06416v1.pdf`, Algorithm 2:

| Step | Operation | Nature |
|---|---|---|
| 3 | $\theta_* = \arg\max_\theta\lvert\eta^{[k]}(\theta)\rvert$ | add one atom |
| 7 | LASSO on amplitudes, positions frozen | convex (FISTA) |
| 8 | joint solve over amplitudes **and** positions | non-convex (bounded BFGS) |
| 9 | remove zero-amplitude atoms | prune |

Stopping: $|\eta^{[k]}(\theta_*)| \le 1$. Step 3 is grid search + Newton.

**Theorem 3.** If $m_{a,x}$ is the *unique* solution and $\eta_\lambda$ is nondegenerate —
$|\eta_\lambda(\theta)|<1$ off support, $\eta_\lambda''(\theta_i)\neq0$ on it — the algorithm
terminates finitely. The paper is explicit that *joint* optimization in step 8 is what buys
this.

**[V\*]** Proved for $d=1$; the multidimensional extension is asserted, not shown.

**[V] The dictionary is translation-only.** Theorem 3 requires $\phi \in \mathrm{KER}(2)$
(Definition 4): $\phi \in C^k(X,H)$, $x\mapsto\langle\phi(x),p\rangle_H$ vanishing at
infinity for all $p$, and $\sup_x\|D^i\phi(x)\|_H < \infty$ for $i\le k$. Throughout, $\phi$
is a **fixed kernel** and $x$ is a **position**; the worked example is
$\phi(x) = e^{-(\cdot-x)^2}$ on $X=[0,1]$. There is no scale or shape parameter anywhere in
the setting.

**[A]** So the "changes for $d\in\mathbb{N}^*$" remark concerns the dimension of *position*,
not the addition of covariance parameters. Applying Theorem 3 to a 5-dimensional
position-plus-covariance dictionary is not an extension of the stated result.

**[A]** $\mathrm{KER}(2)$ also fails at both ends of scale for a Gaussian family with free
covariance: as $\Sigma$ approaches singularity the differentials $\|D^i\varphi\|$ blow up
like $1/s$, and as $\Sigma$ grows large $\varphi_\theta$ tends to a constant on a bounded
domain so $\langle\varphi_\theta,p\rangle$ does not vanish. It can hold on a scale range
bounded away from both extremes.

**Covers:** exact solution of (P$\lambda$), 1D, **fixed kernel, translation-only**, scalar,
$L^2$.
**Does not cover:** free covariance, or scale. One atom per iteration with a full BFGS
re-solve. **[A]** The transferable element is the certificate as densification criterion —
batched, adding many atoms per round at local maxima of $|\eta|$ — not the iteration
schedule or the termination theorem.

### 6.2 Conic Particle Gradient Descent

**[V]** `papers/1907.10300v2.pdf` — Chizat. Discretize the measure into particles, run
non-convex gradient descent on positions and weights:

$$(r_i,\theta_i) \leftarrow \mathrm{Ret}_{(r_i,\theta_i)}\big(-2\alpha r_i J'_\nu(\theta_i),\ -\beta\nabla J'_\nu(\theta_i)\big)$$

$\alpha$ = weight step-size, $\beta$ = position step-size. For $\alpha,\beta>0$ this is
gradient flow in the **Wasserstein–Fisher–Rao** metric; under the mirror retraction the
weight update is $r\exp(\delta r/r)$ — multiplicative — with additive position updates.

**Theorem 4.2.** Under (A1–5), with $\rho$ absolutely continuous with smooth positive
density and $\log\rho$ Lipschitz, if $W_\infty(\nu_0,\rho)\le(J_0-J^\star)/C$,
$\alpha\le(J_0-J^\star)^{1+\epsilon/2}/C$ and $\beta\le(J_0-J^\star)\alpha^2/C'$, then
projected gradient descent converges to the **global** optimum, linearly after $k_0$.
Complexity $\log(1/\epsilon)$ rather than $\epsilon^{-d}$.

**[V\*]** $\nu_0$ is an $m$-sample approximation of $\rho$ with $W_\infty$ rate
$\tilde O(m^{-1/d})$; the paper states this exponential dependence on $d$ is **unavoidable**.
$m>m^\star$ is required, and $J_0-J^\star$ shrinks as problems get harder, forcing $m$ up.

**[A]** $d = \dim\Theta = 5$ for 2D Gaussians, versus far larger for 3DGS, so the
exponential is at least plausibly survivable here.

#### Do (A1–5) hold for the Gaussian dictionary?

| | Statement **[V]** | Verdict for $\Theta=\mathbb{R}^2\times\mathrm{SPD}(2)$ |
|---|---|---|
| **A1** | $\Theta$ a **compact** Riemannian manifold **without boundary**; $\varphi,R$ twice Fréchet differentiable with locally Lipschitz second derivatives | **Fails** |
| **A2** | $h(r)=r^2$, metric $(\alpha(r),\beta(r))=(\alpha,\beta/r^2)$ | Holds by construction — a choice of algorithm |
| **A3** | $R$ convex | **Holds** — square loss named explicitly |
| **A4** | unique global minimizer, finitely supported | Not checkable a priori |
| **A5** | $\nabla^2R(f^\star)\succ0$; $\sigma_{\min}(K)>0$; $\sigma_{\min}(H)>0$; **strict slackness** | Not checkable a priori; §7 argues it is implausible here |

**[A] A1 fails twice.** $\mathbb{R}^2\times\mathrm{SPD}(2)$ is non-compact. The natural
repair — bound the image domain and the scale range — produces a manifold *with* boundary,
which (A1) also excludes. Positions can be compactified to a torus (periodic image, standard
in this literature) and orientation is already $S^1$, but **log-scale has no natural
boundaryless compactification**. That is the genuine obstruction. The smoothness half of A1
holds only on a scale range bounded away from singular and from very large — the same range
$\mathrm{KER}(2)$ needs (§6.1), and the reason GaussianImage's scale lower bound exists.

**[V] A2 is not a hypothesis about the problem** — it fixes $h(r)=r^2$, so mass is
non-negative and $\nu\in\mathcal{M}_+(\Theta)$. Signed measures require $\Theta$ to be
doubled into two copies with opposite-signed $\varphi$; colour compounds this.

**[V] The author's own position on A5:** *"The last two assumptions unfortunately depend on
an a priori unknown object $J'_{\nu^\star}$... In spite of this drawback, the local analysis
leads to interesting qualitative insights on the dynamics in practice."* And: *"it is an open
question to even show local convergence when (A5) does not hold."*

**[V] Regularization is required for the rate.** In the signed case $\lambda>0$ is necessary
for (A5), and $K_\lambda = K_0+o(\lambda)$, $H_\lambda = \lambda H_0+o(\lambda)$, so the local
rate $\kappa_0$ is of order $\lambda$ — **as $\lambda\to0$ the exponential convergence rate
is lost.**

**[A] This is a genuine tension with §1.1.** Small $\lambda$ shrinks the relaxation gap
between (P$\lambda$) and (P0); large $\lambda$ is what buys the fast certified rate. You
cannot have both. Any experiment must state where on that curve it sits.

**[V] Polynomial dependencies.** $(J_0-J^\star)^{-1}$ and $\kappa_0^{-1}$ depend polynomially
on $\lambda$, the regularity of $\varphi$ and $R$, the amplitude ratio
$\max_i r_i/\min_i r_i$, the inverses of $\sigma_{\min}(\nabla^2R(f^\star))$,
$\sigma_{\min}(H)$, $\sigma_{\min}(K)$, and the strict-slackness margin $v^\star$.
**[A]** The amplitude ratio is a practical concern for images, where atom amplitudes span
orders of magnitude.

**Covers:** global convergence of particle gradient descent under (A1–5).
**Does not cover:** this problem, as stated. A1 fails; A4 and A5 are unverifiable and §7
argues A5 is implausible at image atom densities.

### 6.3 Semi-discrete optimal transport

**[V]** `papers/ma.pdf` — Kitagawa, Mérigot & Thibert: damped Newton for semi-discrete OT
with **global linear convergence**, under the Ma–Trudinger–Wang condition on the cost and
quantitatively connected support of the source density. Quadratic cost is covered.

**[V]** `papers/dGBOD12.pdf` — de Goes et al. recast capacity-constrained Voronoi
tessellation as semi-discrete OT. Optimal partitions are **power diagrams** (Laguerre
cells); capacity constraints are enforced exactly via *"a concave maximization w.r.t. the
weights via a step-adaptive Newton method"*, at cost of order a single Newton step. Handles
arbitrary density $\rho$.

**[V]** `papers/Balzer_etal_2009_CCPDAVoLM.pdf`: capacity of a point = area of its Voronoi
region weighted by $\rho$; the constraint is equal capacity.
**[V]** `papers/MSR-TR-2009-174.pdf`: fast CCVT, orders of magnitude faster, scales in point
count.

**Covers:** placing $N$ points to match a *prescribed density*, optimally and provably.
**Does not cover — the objective.** Nobody wants to match a density; they want to minimize
reconstruction error. The density is a surrogate, and it comes from Zador's theorem, which
is about nearest-neighbour quantization, not kernel approximation (§6.4). **The rigour here
is real but attached to a proxy whose relationship to the true objective is unestablished.**
This is a principled *initializer*, not a solved subproblem of (P0).

### 6.4 Covariance and density from local image geometry

**[V]** `papers/BLdG+16.pdf` — Budninskiy et al.: cell complexes with local anisotropy
conforming to the Hessian of a given function, **for any given local mesh density**, built
on approximation theory as an anisotropic extension of CVT. Uses first-type Bregman diagrams
— power diagrams whose sites carry a scalar weight *and* a vector-valued shift.

**[V\*]** Stated for the Hessian of a **convex** function. Image Hessians are indefinite.
**[A]** Standard fix from anisotropic mesh adaptation: use $|H|$ (eigenvalue absolute
values) or the structure tensor, both PSD. Whether the guarantees survive that substitution
is not established.

**[U]** Anisotropic mesh adaptation: the optimal metric for interpolation error is defined by
the modified Hessian; tessellations align with its eigenvectors and anisotropic quotients
match its eigenvalue ratios.

**[U]** Zador: the $N$-point optimal quantizer's empirical distribution converges to density
$\propto\varphi^{d/(d+s)}$; for $d=2$, $s=2$, $\propto\sqrt\varphi$.
**[A]** For nearest-neighbour quantization, not Gaussian-kernel approximation. A design
heuristic to validate empirically, not a theorem about splatting.

### 6.5 EM, and the one direct precedent

**[V]** `papers/gbc.pdf` — Papas et al., *Goal-based Caustics* (2011). Non-negative image
decomposition into overlapping anisotropic Gaussian kernels:

1. sample image intensity densely ($n = 4{,}000{,}000$ points)
2. compute a **CCVT**; take Voronoi centroids as Gaussian centres
3. initialize $\Sigma_i = \mathrm{diag}(\sigma^2,\sigma^2)$ with $\sigma$ = radius of a $k$-NN
   query, $k$ = the region's capacity
4. refine with sparse **EM**

Reported at ~**1024 Gaussians** per image; always anisotropic ("significant qualitative
improvement" over isotropic).

**[V]** They experimented with several initialization strategies and *"obtained the most
reliable results using regular point sets that adapt to the image intensity"* — CCVT won an
empirical initialization comparison for this exact problem.

**Covers:** an existence proof that the CCVT + EM pipeline works at ~10³ atoms.
**Does not cover:** any optimality claim. EM converges monotonically to a local optimum only,
which the paper states directly.

### 6.6 Saddle-escape splitting

**[U]** SteepGS: necessary conditions for densification to reduce loss; **minimal offspring
count is exactly 2**; optimal displacement along the minimal eigenvector of a per-primitive
splitting matrix when its eigenvalue is negative; analytic offspring-weight normalization.
Reported ~50% fewer primitives at equal quality.

**[A]** Derived for the 3DGS objective *with* $\alpha$-blending — i.e. for the case §2.1
argues the measure theory does not cover. The derivation is primitive-local so it plausibly
transfers, but it is not a result about (P$\lambda$).

---

## 7. The regime mismatch

The BLASSO literature is built for **sparse, well-separated** atoms — the super-resolution
regime, a few point sources far apart. Image encoding is the opposite: $10^4$–$10^5$ heavily
overlapping atoms tiling a bounded domain. Nearly every guarantee beyond the certificate is
conditioned on separation, and image encoding does not satisfy it.

**[V] The separation condition, made concrete.** `papers/2509.12889v5.pdf` — the one result
covering unknown covariances — states recovery for mixtures whose components are separated in
a semi-distance

$$d(x,x')^2 = \sum_k \left[ \frac{(t_k-t'_k)^2}{u_k'^2+u_k^2+\tau^2} + \ln\!\left(\frac{u_k'^2+u_k^2+\tau^2}{\sqrt2\,u_k}\right)\right]$$

with $t$ the means and $u$ the standard deviations. The first term is centre separation
normalized by **combined width**. Guarantees are stated on "near regions" — balls of radius
$r_e$ in this semi-distance around each true component — versus a complementary "far region".

**[A] Two atoms are at semi-distance of order 1 when their centres differ by roughly their
combined width.** A dense image tiling places neighbouring Gaussians exactly there, by
construction: that is what tiling *is*. So image encoding sits at the bottom of the
separation scale, not above the threshold the theorems require.

**What this invalidates at image densities:**

- exact support recovery — the claim that BLASSO finds "the right" atoms
- SFW finite termination, which needs a unique solution and a nondegenerate certificate
- (A4) uniqueness — implausible when overlapping atoms can trade mass
- **(A5) strict slackness** — requires $J'_{\nu^\star}$ to vanish *only* at support points
  with margin $v^\star$. With densely packed overlapping atoms $J'_{\nu^\star}$ is near zero
  across large regions of $\Theta$, $v^\star\to0$, and $\kappa_0^{-1}$ depends polynomially
  on $1/v^\star$
- the assumption in §1.1 that the relaxation gap is small — provably zero in the separated
  regime, **unknown and potentially large here**

**What survives:** the certificate and the duality-gap bound (§5), because they are convex
duality and need no separation.

**[A] The consequence for framing.** This is not "BLASSO recovers the true Gaussians of an
image" — images have no true Gaussians, and the recovery theory does not apply at these
densities anyway. It is: *(P$\lambda$) is a convex objective whose optimum can be certified,
and which happens to have Gaussians as its atoms.* Optimality of the encoding, not recovery
of a signal. That is still worth having, and it is a narrower claim than the literature might
suggest.

## 8. The guarantees do not compose

§6 lists five results with convergence or optimality guarantees. They are guarantees **for
five different problems**, and there is no argument here that they compose:

| Result | Guarantees what | For which problem | Applies here? |
|---|---|---|---|
| BLASSO certificate | global optimality test | any bounded linear $\Phi$, $L^2$, scalar | **yes** |
| FW duality gap | suboptimality bound | any bounded linear $\Phi$ | **yes** |
| SFW Theorem 3 | finite termination | $d=1$, **fixed kernel, translation-only** | no |
| CPGD Theorem 4.2 | global convergence | (A1–5) | **no — A1 fails** |
| KMT damped Newton | global linear convergence | density matching, not (P0) | different objective |
| SteepGS split rule | optimal offspring count | 3DGS objective **with** $\alpha$-blending | different objective |

Only the first two apply. The rest are motivation.

**What would be needed to make the composition real**, roughly in order of tractability:

1. **A boundaryless compactification of the scale dimension**, or an extension of CPGD's
   analysis to manifolds with boundary (§6.2). Positions can be a torus and orientation is
   already $S^1$; log-scale is the obstruction. Without this, §9 is motivation, not
   implication.
2. Bound the relaxation gap between (P$\lambda$) and (P0) empirically (§11 E2), at a stated
   $\lambda$ — noting that $\lambda\to0$ shrinks the gap and destroys the convergence rate
   (§6.2).
3. Establish whether anything survives in the **dense, overlapping** regime (§7). This is the
   deepest question here and it may simply be open: the literature has not needed it, because
   super-resolution never operates there.
4. Extend SFW's termination argument to dictionaries with free covariance, or accept it as
   heuristic.
5. Relate §6.3's density-matching objective to reconstruction error, or leave it permanently
   as initialization.
6. Establish the vector-valued (colour) case (§2.3).

Items 1 and 3 decide whether this is a research programme or a well-referenced heuristic
built around one genuine tool.

---

## 9. Implications, conditional on §§7–8

**[A] Status of everything below.** CPGD's (A1) fails for this parameter space (§6.2) and its
(A5) is implausible at image densities (§7), so I1–I3 are **not implied by the theorem** —
they are what the analysis suggests if the obstructions can be repaired. They remain worth
testing, because each is cheap and each makes a concrete prediction that could fail. Read
them as hypotheses, not consequences. I4 is different: it follows from §5 alone.

**I1 — Initialization should have full support. *(moderate)*** CPGD requires $\nu_0$
$W_\infty$-close to a reference $\rho$ with smooth positive density over $\Theta$, densely
sampled, with $m>m^\star$.

This cuts against content-adaptive initialization: concentrating atoms where image gradients
are large gives up the full-support condition. But $\rho$ is the *prior on the solution* and
smaller $\bar H(\nu^\star,\rho)$ improves the bounds, so the indicated design is a
geometry-informed $\rho$ that remains **strictly positive everywhere** — neither uniform nor
concentrated.

**[A] Caveat against overreading this.** Uniform-random initialization does not satisfy the
hypothesis either. **[V]** GaussianImage initializes covariance parameters and colour
coefficients from a uniform distribution and positions as $\mu = \mathrm{atanh}(2\,\mathrm{rand}(2)-1)$,
and imposes a lower bound on the scaling elements to stop covariances collapsing.

**[A]** Two problems. First, the uniform is over a *chart* — either Cholesky factors or
$(R,S)$ rotation–scaling, and **[V]** GaussianImage offers both and notes the decomposition
is not unique. The pushforward of a uniform density on Cholesky coordinates is not the
pushforward of a uniform on $(R,S)$, and neither is a canonical measure on $\mathrm{SPD}(2)$.
The reference measure $\rho$ is therefore an artefact of parameterization. Second, the scale
lower bound truncates $\Theta$, which helps with non-compactness but is not the same as
sampling $\rho$ densely.

So no existing initialization satisfies the hypothesis, and *which chart you parameterize
covariance in silently determines $\rho$*. That is a concrete, checkable design question
that current work does not appear to treat as one.

**I2 — Weight updates should be multiplicative. *(moderate)*** The Fisher-Rao component acts
multiplicatively on mass. Additive Adam on amplitudes does not implement this. Directly
testable.

**I3 — Position and weight step-sizes should be coupled through the atom scale.
*(moderate)*** Two distinct statements, and they are not the same rule.

**[V] Globally**, Theorem 4.1 requires $\beta/\alpha$ small, *"at least in the early stages
of the algorithm"*.

**[V] Locally**, the paper explicitly disclaims the reading that $\alpha=\beta$ is best —
*"just an artefact of the way the upper bounds are presented"* — and gives a conditioning
heuristic instead:

$$2\alpha\|\varphi\|_\infty \approx \beta\,\mathrm{Lip}(\varphi)$$

**[A]** For a Gaussian atom of width $s$, $\|\varphi\|_\infty$ is scale-free while
$\mathrm{Lip}(\varphi)$ — the sensitivity of $\varphi_\theta$ in $L^2$ to moving the centre —
grows like $1/s$. So the heuristic gives $\beta/\alpha \propto s$: **the position step-size
should scale with the atom's spatial extent.** This is not "positions move slowly"; it is a
per-atom, scale-dependent coupling.

**[A]** 3DGS scales its position learning rate by scene extent, which is the same idea
arrived at empirically. Worth checking whether a *per-atom* version — step-size proportional
to each Gaussian's own scale rather than a global constant — outperforms the global one.
That is a cheap, self-contained experiment and the most directly actionable item here.

**I4 — Densify at $\arg\max_\theta|\eta(\theta)|$, searching covariance as well as position.
*(high — this follows from §5 directly, not from CPGD)***

**I5 — Split into exactly 2 along the minimal eigenvector. *(low)*** Rests on **[U]** and on
a derivation for a different objective (§6.6).

**I6 — Consider quasi-Newton for the joint update. *(low)*** SFW uses bounded BFGS rather
than SGD for step 8. Suggestive only.

**A note on framing densification.** WFR flow has a transport component (moving mass) and a
Fisher-Rao component (creating and destroying it), and it is tempting to read clone/split
and prune as a discretization of the latter. **[A] This is probably only partly right.**
Fisher-Rao alters mass at fixed support, whereas clone/split alters the *support* by
duplication with perturbation. MCMC-style relocation is structurally closer to the
Fisher-Rao operation than clone/split is. Treat the analogy as motivation for looking at
relocation schemes, not as an explanation of why densification works.

---

## 10. Open questions requiring measurement

None of these have numbers yet, and several gate the programme.

- **Cost of the certificate.** Each densification round needs $\arg\max_\theta|\eta(\theta)|$
  over a 5-dimensional $\Theta$. SFW does grid search + Newton with grid size tied to the
  operator's bandwidth. Unmeasured here, and it determines whether I4 is practical.
- **Cost of OT initialization** at $N = 10^4$–$10^5$ on 2D images. Fast CCVT is reported to
  scale well **[V]** but no timings were extracted.
- **$\lambda \to N$.** To hit a target atom count you sweep $\lambda$. The stability and cost
  of that path, and whether warm-started homotopy works here, is unknown.
- **Size of the relaxation gap** (§1.1), and how it varies with $\lambda$ given that
  $\lambda\to0$ shrinks it while destroying the convergence rate (§6.2).
- **Whether anything survives the dense regime** (§7), and whether the scale dimension can be compactified (§8 item 1).
- **Perceptual acceptability of $L^2$-optimal encodings** at low atom counts (§2.2).

---

## 11. Programme

Ordered so that the results that could invalidate the rest come first.

**E1 — Certified suboptimality under (P$\lambda$).** Fix $L^2$, single channel. Run a BLASSO
solver to a certified duality gap. Separately, evaluate existing encoders' outputs *under the
same (P$\lambda$) objective*. Report the certified distance to optimum for each. Well-posed
because everything is scored on one objective; note that existing encoders are not trying to
minimize it, so this measures representational distance, not encoder failure.

**E2 — The relaxation gap.** On instances small enough for near-exhaustive search over (P0)
— tiny images, $N$ of order 10 — compare the best (P0) solution found to the (P$\lambda$)
solution at matched support size. Isolates the $\ell_1/\ell_0$ gap from encoder
suboptimality, which E1 cannot. Small, and it decides how much E1 is worth.

**E3 — Certificate-driven densification.** Batched $\arg\max_\theta|\eta(\theta)|$ over a
coarse covariance grid, replacing error-magnitude sampling. Compare at equal $N$ and equal
compute against existing densifiers. Tests I4, the highest-confidence implication.

**E4 — Geometry-informed initialization.** Structure tensor or $|H|$ field → density law →
semi-discrete OT placement → Hessian-derived covariances, kept strictly positive everywhere
per I1. Compare against uniform-random and gradient-guided initialization at equal $N$ and
equal iteration budget.

**E5 — Per-atom step-size coupling.** Test I3's $\beta/\alpha \propto s$ against a global
position learning rate, at fixed atom count and iteration budget. Self-contained, no new
machinery, and the theory makes a specific prediction that can fail.

**E6 — Test the structural prediction (§3.2).** For atoms near edges, regress
$\log(\text{minor axis})$ on $\log(\text{major axis})$. Parabolic scaling predicts slope $2$.
Falsification: slope significantly different from 2 in a well-converged solution means either
the optimizer is not reaching the regime the theory describes, or the images are not
cartoon-like. Either answer is informative and the test is cheap.

**E7 — CPGD-conformant optimizer.** Multiplicative weight updates, tuned $\beta/\alpha$,
overparameterized start. Only worth building after §8 item 1 is settled.

### Reference points

**[V]** `papers/tip2006.pdf` — Figueras i Ventura, Vandergheynst & Frossard (2006): matching
pursuit over translated, rotated, anisotropically-scaled atoms, reported as comparable to
JPEG2000 and SPIHT at low rates.

**[V\*]** Its dictionary is *richer* than the splatting one: the primary atom is a Gaussian
along one axis × second derivative of a Gaussian in the orthogonal axis, with a separate
pure-Gaussian sub-dictionary. Erb–Hangelbroek–Ron explicitly place such modulated atoms
outside $\mathcal{D}$.

**[A]** Its results are reported on a bits axis, not an atom-count axis, so they do not
transfer to (P0) without atom counts that the paper does not appear to report. Useful as
evidence that greedy certificate-like encoders over this atom family were competitive in
2006; not usable as a numerical baseline here.

**[V]** `papers/gbc.pdf` gives the one atom-axis data point available: ~1024 anisotropic
Gaussians for a recognizable natural image under a non-negative, single-channel objective.

---

## Appendix: sources

PDFs in `papers/`; citations in `papers/README.md`.

| Tag | File | Used for |
|---|---|---|
| **[V]** | `1910.10319v1.pdf` | dictionary definition, $N$-term rates, cartoon class |
| **[V]** | `1811.06416v1.pdf` | BLASSO, dual certificate, SFW, finite termination |
| **[V]** | `1907.10300v2.pdf` | CPGD, WFR flow, Theorem 4.2, $\alpha/\beta$ |
| **[V]** | `1805.09545v2.pdf` | mean-field / OT framework |
| **[V]** | `2509.12889v5.pdf` | BLASSO for GMM, unknown **diagonal** covariances only |
| **[V]** | `ma.pdf` | semi-discrete OT, damped Newton, global linear convergence |
| **[V]** | `dGBOD12.pdf` | CCVT as OT, power diagrams, concave maximization |
| **[V]** | `Balzer_etal_2009_CCPDAVoLM.pdf` | CCVT definition |
| **[V]** | `MSR-TR-2009-174.pdf` | fast CCVT |
| **[V]** | `BLdG+16.pdf` | Hessian-based anisotropy, Bregman diagrams |
| **[V]** | `gbc.pdf` | CCVT + EM precedent, ~1024 kernels, initialization comparison |
| **[V]** | `tip2006.pdf` | matching pursuit reference, atom definition |
| **[V]** | `2403.08551v5.pdf` | Eq. 7 linearity, $L^2$ loss, no density control, initialization |

**Not read.** SteepGS; Zador / quantization theory; anisotropic mesh adaptation;
sparse-approximation hardness; group BLASSO; Structure-Guided Allocation; Image-GS. All
**[U]**.

**Closest theoretical result to the target problem.** **[V]**
`papers/2509.12889v5.pdf` extends BLASSO to Gaussian mixtures with component-specific unknown
**diagonal** covariances, with non-asymptotic recovery for means, covariances and weights via
a Fisher-Rao semi-distance and an explicit separation condition. Two gaps: rotation is not
covered, and the setting is density estimation from i.i.d. samples rather than $L^2$
approximation of a signal.

### Literature search note

"2D Gaussian Splatting" is ambiguous. Most results — including nearly all robotics, satellite
and medical work — refer to 2D Gaussian *surfels embedded in 3D* (Huang et al., SIGGRAPH
2024), a different problem.
