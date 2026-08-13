# Optimal encoding of images as 2D Gaussians

Given a target image, find the best representation as a set of 2D Gaussians, and know how far
from best any given representation is.

Not a compression document — rate, entropy coding and decode speed are out of scope.

**Status tags.** **[V]** verified — either against a primary source in `papers/` or measured by
an experiment in `experiments/`; the two are distinguished in the text, and a **[V]** never
means "seems right" · **[V\*]** verified with an inline caveat · **[U]** unverified, search
summaries only · **[A]** analysis from **[V]** facts · **[A←U]** analysis resting on an
unverified premise. Methodology rules and this document's error history are in Appendix A.

---

## 1. Summary

**The question.** (P0) is non-convex: it has $N!$ equivalent minima, saddles between them, and
local optima a descent method can get stuck in. The hope motivating this document is that lifting
it to a measure — (P$\lambda$) — makes it *convex*, so that any minimum found is the global one
and the certificate of §6 proves it. The question is whether that hope survives contact with the
actual problem.

**The answer, in one line.** The lifting is genuinely convex and the certificate genuinely works,
but the convex problem is not (P0) and **provably cannot be made into it** (§2.2) — so what you
reach globally is the optimum of the wrong problem, and §10 measures it landing further from (P0)
than a cheap non-convex method does.

**The central negative result (§10).** On all six certified rows, **greedy** produced a better
$N$-atom encoding than the convex BLASSO route, even after debiasing and full polish. Adding an
$\ell_1$ term buys a certificate and costs reconstruction quality. **[V] §10.4 shows this is the
formulation and not the solver**, by shrinking the problem until both sides are solved exactly.

*Terminology:* "greedy" here means **certificate-driven greedy at $\lambda=0$** — it places each
new atom at $\arg\max_\theta|\eta(\theta)|$, so it uses the dual certificate as a *placement
criterion*, but it solves no convex problem and yields **no optimality certificate**. It is also
roughly an order of magnitude cheaper than the BLASSO route, which repeats a comparable inner
loop across a $\lambda$ bisection. This is matching pursuit with continuous refinement.

**What that means for the framing.** (P$\lambda$) is *not* justified as the working definition
of optimal encoding on quality grounds. Its defensible role is narrower: a **diagnostic
instrument** telling you where a solution sits relative to a computable optimum, not a method
for producing the best solution.

**The second negative result (§10.2, §10.3).** Greedy is not thereby shown to be good. A
certified lower bound on the (P0) optimum — the first absolute measurement here on targets
outside the model class — goes **vacuous by $N=4$–8**, at or below the budgets §10 itself
reports. Where it does bite, at $N=1$, greedy turns out to be *exactly* optimal, and the bound's
inability to see that (it certifies only 24–34%) is the honest measure of how blunt the
instrument is. Separately, on a dictionary small enough to enumerate exhaustively, greedy's
placement is **provably suboptimal by 8.5%** at $N=2$ on the cartoon target. So §10 remains a
statement about the better of two methods, and how good either is in absolute terms is still
unknown where it matters.

**What survived scrutiny:**

| | status |
|---|---|
| Greedy beats random restarts | **[V]** 9/9 rows, §10.1 |
| Greedy beats BLASSO at matched $N$ | **[V]** 6/6 *certified* rows, §10 (3 more agree but BLASSO did not converge there) |
| **Convexifying over measures cannot encode $N$** | **[A]** proof, §2.2 — the gap is structural and no convex penalty removes it |
| That ordering is the formulation, not the solver | **[V]** §10.4, both sides solved exactly: $\ell_1$ is 6–409% worse than $\ell_0$ at matched $N$ |
| Dictionary structure helps on-grid, washes out off-grid | **[V]** §10.5 — ~25% on-grid, a few % refined, reversed at 64 splats |
| Wavelet placement does not work | **[V]** §10.6 — loses to greedy 12/12, and to *random* 10/12 |
| **No certified global optimum is reachable by any route tested** | **[V]** §10.8, §10.9 — big-M and perspective relaxations both fail; §2.2 rules out the convex-measure route by proof |
| But the optimum is often *found* anyway | **[V]** §10.8 — greedy+swap attains the exhaustively verified optimum in 4/8 cells, and at $N{=}4$ matches enumeration over 153.8M supports |
| How far greedy is from optimal | **[V]** exactly optimal at $N{=}1$; bound vacuous by $N{=}4$–8 — §10.2. **Open at §10's own budgets** |
| Greedy's placement is itself suboptimal | **[V]** 8.5% at $N{=}2$ on cartoon, 0.5% on face, 0% on two others — §10.3. *Grid* greedy; refinement may repair some of it |
| The optimality certificate is computable and self-diagnosing | **[V]** §6, §9.4 |
| TV needs norm-weighting over a scale-varying dictionary | **[V]** §7.1.1 |
| The analytic norm is exact in the interior, 3.3× off at the frame | **[V]** §7.1.2 |
| Forward model linear, loss $L^2$ — preconditions hold | **[V]** §3 |
| Relaxation gap non-monotone in separation | **[V]** §9.1; mechanism untested |
| Curvelet-optimal rates are *achievable* by the dictionary (existence) | **[V]** §4; **never tested empirically** |

**What is not established.** Everything measured is at $N\le16$ atoms on $48^2$–$192^2$ images.
Real encoding is $10^3$–$10^5$ atoms on $10^6$ pixels — three to four orders of magnitude away.
No conclusion below is known to survive that gap, and §9.3 shows the solver cannot currently
reach it. §11 lists every open question with the experiment that would settle it.

**[A] One methodological gain.** §10.2's bound is valid for *any* dual point, so it needs no
solver to converge. It is the only instrument here that **bounds the optimum** without one —
§10.1 and §10.3 also run free of §9.3's ceiling, but they compare methods rather than bound
anything — and so the only route to an absolute statement at realistic $N$ that does not wait
on U5.

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

**[A] A second, unrelated relaxation appears in §10.2** — mass-constrained rather than
$\lambda$-indexed, and free of any atom count. Its slack is a different quantity from the gaps
above, measured against a different reference, and the two are not comparable. Nothing here
relates them; they share only the word.

### 2.2 Why the gap cannot be removed — convexification annihilates $N$

**[A] Elementary, from standard facts, and it decides the framing.** The extreme points of the
total-variation ball $\{m : |m|(\Theta)\le\tau\}$ are the signed Diracs $\pm\tau\delta_\theta$,
which are **one**-atom measures. The ball is convex and weak-\* compact, so by Krein–Milman it is
the closed convex hull of those extreme points. Hence for **every** $N\ge1$:

$$\overline{\operatorname{conv}}\,\{m : m \text{ has} \le N \text{ atoms},\ |m|(\Theta)\le\tau\} \;=\; \{m : |m|(\Theta)\le\tau\}$$

The atom count is annihilated. **No convex program over $\mathcal{M}(\Theta)$ can distinguish
"$N$ atoms" from "any number of atoms of the same total mass"** — not $\ell_1$, not a reweighted
or spatially varying penalty, not any convex functional whatever, because they all optimize over
the same convex set and that set has already forgotten $N$.

**[A] Three consequences.**

1. §2.1's relaxation gap is **structural**, not a defect of the penalty. Choosing a better convex
   regularizer cannot shrink it. §10.4 measures what it costs with every solver exact.
2. This is also why §10.2's bound is intrinsically loose: it bounds a mass-constrained
   relaxation, which is the tightest thing any measure-space convexification can see.
3. The hope in §1 fails at the formulation, not the implementation. §9.3's certification ceiling
   and U5 are real engineering problems, but fixing them would deliver the global optimum of
   (P$\lambda$), which §10.4 shows is the wrong target.

**[A] The positive half, and it points somewhere concrete.** On a **finite** dictionary with
bounded amplitudes the picture reverses:

$$\operatorname{conv}\{c : \|c\|_0\le N,\ \|c\|_\infty\le M\} \;=\; \{c : \|c\|_1\le NM\}\cap\{c:\|c\|_\infty\le M\}$$

which **does** depend on $N$. That is exactly the structure exact-$\ell_0$ branch-and-bound
exploits, and why solvers in the sparse-regression literature reach $p\sim10^7$ variables at
~20 non-zeros. **[A]** So the grid is not merely a computational convenience: by imposing a
minimum separation it is what makes sparsity convex-representable at all. §8 says the *recovery*
theory needs separation; this says the *optimization* needs it too, for an unrelated reason.
**[U]** The branch-and-bound literature is cited from search summaries, not read — §11 U19.

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

#### 7.1.2 The analytic norm is exact in the interior and wrong at the frame

**[V] Measured** (`experiments/results/e3_norm.txt`, 6000 draws from E3's truncated $\Theta$ at
$n=64$). The implementation divides out the *continuum* norm $\|\varphi_\theta\|^2=n^2\pi
e^{-(u+w)}$. Against that analytic value of 113.4, the actual discrete grid norms run
min 34.3, median 112.6, max 113.4 — a **3.30× spread**. Rank correlations of the norm ratio
against candidate causes:

| edge distance | major axis | minor axis | shear $\lvert v\rvert$ |
|---|---|---|---|
| $+0.684$ | $-0.587$ | $-0.395$ | $-0.082$ |

Conditioning on atoms whose support clears the image edge ($\text{edge}>2\times$ major axis)
collapses the spread to **1.00×**. Conditioning on a resolved minor axis alone does not (3.30×).
(`e3.txt` reports 3.15× for the same quantity — a different random sample of $\Theta$, not a
different result.)

**[A]** So the analytic norm is *exact* on the pixel grid wherever the atom is not clipped, and
the entire deviation is boundary truncation of wide atoms near the frame, which are
under-weighted by up to 3×. Note $\det M$ is independent of $v$, so shear cannot enter the
analytic norm at all — and indeed does not enter the measured one. The commensurate regularizer
of §7.1.1 is therefore right in the interior and wrong at the border.

**[A]** It does not follow that this costs anything. §10.2 found greedy attaining the *exact*
$N=1$ optimum on all four targets despite the spread, so the non-uniformity does not bite at the
argmax there. Whether it bites anywhere else is untested — §11 U17. The fix, if one is needed,
is to divide by the measured discrete norm rather than the analytic one.

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

| script | what it measures | sections |
|---|---|---|
| `e2_relaxation_gap.py` | in-model targets, exact optimum | §9.1 |
| `e2b_natural.py` | out-of-model targets, bounded optimum | §9.2–9.5 |
| `e3_absolute_bound.py` | certified lower bound on the (P0) optimum; exhaustive $N{=}2$ myopia; dictionary norm attribution | §10.2, §10.3, §7.1.2 |
| `e4_exact_l0.py` | exact $\ell_0$ vs exact $\ell_1$ with **no solver error on either side**; coherence sweep; dictionary design | §10.4, §10.5 |
| `e5_dictionary_scaling.py` | dictionary comparison budgeted in **splats**, to 64 splats, with off-grid refinement | §10.5 |
| `e6_wavelet_init.py` | wavelet transform as a replacement for the placement search | §10.6 |
| `e7_frequency_continuation.py` | Parseval check; frequency continuation vs direct refinement | §10.7 |

**[A] Reproducibility.** Only E3 commits its raw output (`experiments/results/`) and records its
parameters there. E2 and E2b do not, and their committed defaults do **not** reproduce the tables
below: `run()` in E2 defaults to $K=9$ and $r\in\{6,4,3,2,1.5,1\}$, whereas §9.1 reports $K=4$ and
$r\in\{30,15,8,4,2,1\}$; §7.1.1 cites a 192px image and neither script defaults to that (E2: 96,
E2b: 64), so it is not even recorded which of the two produced it. The §9 numbers therefore
cannot currently be regenerated from the repository without guessing. §11 U18.

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

#### 9.1.1 Replicated, and the column that was missing (U18)

**[V]** The table above is not reproducible — its parameters were never recorded (§9 preamble).
So it was re-run on a fresh instance at parameters written down in full
(`results/u18_e2.txt`: $n=128$, $K=4$, $u_{px}=4$, `n_restarts=2`, `seed=0`, every ratio verified
in-frame by the new guard):

| $r$ (widths) | 30 | 15 | 8 | 4 | 2 | 1 |
|---|---|---|---|---|---|---|
| BL debiased, original | 0.00 | 0.00 | 0.00 | **22.86** | 4.70 | 0.38 |
| BL debiased, re-run | 0.56 | 0.00 | 0.00 | **20.53** | 3.23 | 0.19 |
| **BL polished, re-run** | 0.001 | 0.000 | 0.000 | **2.305** | 0.718 | 0.100 |

**[V] The non-monotone hump replicates** — a different instance, different parameters, peak in the
same place and within 10% of the same height. §9.1's central observation is therefore a
reproducible fact and not an artefact of one unrecorded configuration.

**[V] But the polished row changes how the peak should be read, and it was computed all along.**
`e2` prints a BLpolish column that the original table omitted. Local refinement from BLASSO's own
support cuts the $r=4$ figure from 20.53% to **2.305%** — an 89% reduction. **[A]** So most of the
hump is *placement imprecision* that a local solve repairs, not a support chosen in the wrong
basins. The residual 2.3% against a true optimum of exactly 0 is the part that is genuinely
support selection. §9.4's first claim survives — a certified gap of 0.0014% beside 20.53% error
does show the $\ell_1$ solution is far from the $\ell_0$ optimum *in value* — but "genuinely far"
should be read as far in value and close in support.

**[A]** Note also that the re-run gives 0.56% at $r=30$ where the original gives exactly 0.00. At
$r=30$ the corner atoms sit about one $\sigma$ from the frame and are partly clipped, so exact
recovery is not expected. An exact zero there is what one would see if the atoms were *entirely*
off-image — which is the failure §9's preamble describes and which the new guard now refuses to
report.

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

**[A] But "matching pursuit is the encoder" is a comparative claim, not an endorsement.** §10.2
cannot certify greedy at the budgets used here, and §10.3 exhibits a target where greedy's
placement is suboptimal by 8.5% at $N=2$ on a dictionary where the optimum is computable. Both
are consistent with greedy being the best available method *and* being well short of optimal.
Nothing in this document distinguishes those two readings at $N\ge8$.

**[A] What would overturn this.** The comparison is at $N\le32$ on small images, one instance
per cell, with a BLASSO solver having known defects (§9.3). A better-engineered BLASSO at
realistic $N$ could plausibly close or reverse the gap. §11 U7.

### 10.2 How far greedy is from optimal — bounded at small $N$, open at §10's own budgets

`experiments/e3_absolute_bound.py` Leg A; raw output `experiments/results/e3.txt`.

§10 is a *relative* result. This is the first absolute one **outside the model class** — §9.1 is
already absolute, but only because an exact $K$-atom target has a known optimum of 0, which no
photograph does.

**The instrument.** For any $p\in H$ and any $m$ with $|m|(\Theta)\le M$, two elementary
inequalities — $\tfrac12\|r\|^2\ge\langle p,r\rangle-\tfrac12\|p\|^2$ (which is just
$\tfrac12\|r-p\|^2\ge0$) and Hölder on $\langle\Phi^*p,m\rangle$ — compose to

$$\tfrac12\|y-\Phi m\|^2 \;\ge\; \langle p,y\rangle-\tfrac12\|p\|^2-M\,s(p), \qquad s(p)=\sup_{\theta\in\Theta}|\langle\varphi_\theta,p\rangle|$$

Along a ray $p=tq$ this is a concave quadratic in $t$, maximized in closed form at
$L(q;M)=\big(\langle q,y\rangle-M\,s(q)\big)_+^2/\big(2\|q\|^2\big)$. Each candidate direction
therefore costs three scalars, one of them the $\eta$ supremum already implemented for §6.

**[A] Three properties, all from the derivation.** It holds for *every* $p$, so no solver need
converge — this is why §9.3's ceiling does not gate it. The maximum over any family of
directions is still a valid bound, so directions can only help. And $L$ is concave in $p$, so
searching for a good direction is itself a concave maximization rather than a heuristic.

**What it bounds, exactly.** $L(M)\le E_{\rm relax}(M)\le E_{N\text{-atom}}(M)$, where
$E_{\rm relax}(M)$ ranges over measures of mass $\le M$ with **any** number of atoms. The
constraint is on total mass, not on atom count, so this bounds a *relaxation* of (P0) and its
slack is at least $E_{N\text{-atom}}(M)-E_{\rm relax}(M)$.

**[A] That is not §2.1's relaxation gap and the two must not be compared.** §2.1 measures
BLASSO at matched $N$ against the (P0) optimum, a $\lambda$-indexed object. This one is
mass-constrained and atom-count-free. They are different quantities with different units of
comparison, and no result here relates them.

It is a lower bound, so a large gap below greedy is **inconclusive** — it may be slack rather
than suboptimality. That asymmetry is why the next paragraph exists.

**[V] The slack is measured, not assumed.** At $N=1$ the (P0) optimum is computable outright as
$\max_\theta\langle\varphi_\theta,y\rangle^2/\|\varphi_\theta\|^2$, and greedy attained it
**exactly** on all four targets ($+0.0000$pp) — so matched-filter placement is optimal at
$N=1$ despite §7.1.2's norm defect. The bound nonetheless reports 23.7–33.9% suboptimality
there. That figure is therefore *pure slack*, measured against a known answer.

**[A] It does not follow that the slack elsewhere is 24–34%.** Slack varies with $M$ and with
$N$, and nothing here measures it at $N>1$, where no known optimum exists to measure against.
The $N=1$ column establishes that this instrument is blunt, not how blunt it is further down the
table. Read every row below as an upper bound on suboptimality that is loose by an unmeasured
amount.

Certified suboptimality bound, as % of greedy's own error (— = vacuous):

| target | $N{=}1$ | 2 | 3 | 4 | 6 | 8, 12, 16 |
|---|---|---|---|---|---|---|
| cartoon | 33.9 | 71.3 | 89.7 | — | — | — |
| face | 23.8 | 42.2 | 93.3 | 91.8 | — | — |
| ascent | 23.7 | 59.4 | 83.8 | 91.4 | 98.3 | — |
| in-model | 26.0 | 52.0 | 78.0 | *exact* | *exact* | *exact* |

In-model rows from $N=4$: greedy recovers the 4-atom target exactly, so $E_{\rm opt}=0=L$ and
the bound is tight there rather than vacuous.

**[V] The bound goes vacuous at $N=4$ (cartoon), 6 (face), 8 (ascent)** — at or below the
$N\in\{8,16\}$ at which §10 reports. **[A] So U14 is answered only at budgets smaller than the
ones the central result uses.** Nothing here bounds greedy at §10's budgets, and nothing
approaches $10^3$–$10^5$.

**[V] Whether the looseness is intrinsic is target-dependent.** Supergradient ascent on $p$
(18 steps per budget, every iterate certified separately) improved the bound by 0.0% on the
in-model target and on cartoon, but by 2.5–58.5% on ascent and 6.9–1339% on face. **[A]** On the
first two the direction family was already at this instrument's ceiling, so what remains is
relaxation slack; on the other two the search was *not* saturated and no such attribution is
available. An unconditional version of this claim was emitted by the experiment's own log and
is withdrawn — see Appendix A.

**[A] The result is conditional on the mass budget.** The bound covers encodings of total mass
$\le M$, evaluated at $M=M_g$, greedy's own. A better $N$-atom solution spending more mass is not
excluded, and the mass sweep shows the bound is already vacuous at $1.5\,M_g$ for $N\ge2$ on
cartoon and $N\ge3$ on ascent. Every figure above must be quoted with that condition. §11 U15.

**Validity.** The one failure mode that would *invalidate* rather than weaken is an
underestimated $s(q)$ — the same failure §9.4 caught as a negative duality gap. Guards:
positions searched exactly by FFT, a 320-shape bank, a uniform random probe of $\Theta$,
projected Nelder–Mead refinement, an inflation sweep, and two hard checks.

- **C1** — $L(M_g)\le E_{\rm greedy}$ is forced, since greedy's own solution has mass $M_g$.
  Passed on all 32 rows.
- **C2** — in-model, ground truth attains error 0 at mass $M_{\rm true}$, so $L(M_{\rm true})\le0$
  is forced. Near-equal amplitudes make this *nearly tight* rather than slack, since
  $\|y\|^2\le M_{\rm true}s(y)$ becomes an equality at orthogonality with equal amplitudes.
  Passed exactly.

**[A]** Under a 25% inflation of $s$ the bound stays positive only at $N\le2$ (cartoon, face) and
$N\le3$ (ascent, in-model). So the conclusions tolerate a modest supremum error, not a large one.

### 10.3 Greedy's placement is measurably myopic

`experiments/e3_absolute_bound.py` Leg B.

On a dictionary small enough to enumerate — 13 824 atoms (24 shapes × 576 positions, $48^2$
image) — best-2 has a closed form for every pair, so **all 95 544 576 pairs** were evaluated
exactly. Greedy-on-grid fixes $\arg\max|\langle\varphi,y\rangle|$ and then takes its best
partner. The dictionary is identical and there is no continuous refinement, so the difference
isolates the cost of committing to the first atom by matched filter.

| target | greedy best-2 | exhaustive, mass-matched | myopia cost |
|---|---|---|---|
| in-model | 53.161 | 53.161 | **0.000%** |
| ascent | 56.620 | 56.620 | **0.000%** |
| face | 55.505 | 55.243 | **0.476%** |
| cartoon | 42.982 | 39.613 | **8.504%** |

Errors as % of $\tfrac12\|y\|^2$; cost as % of the exhaustive optimum.

**[A] Not an artifact.** Two near-duplicate atoms with large opposing amplitudes reduce error
arbitrarily as their coherence $\to1$ — real arithmetic, useless as an encoding, and enough to
let "exhaustive" win on a technicality. The search was therefore run three ways: unconstrained,
coherence-capped at 0.9, and mass-matched to greedy's own pair. The tabulated column is the
mass-matched one, and the winning pairs sit at coherence 0.00–0.24, so the effect survives
precisely the constraint that would kill an artifact.

**[V] Greedy's placement is provably suboptimal on the cartoon**, by 8.5% at $N=2$. **[A]** This
is the first direct evidence here that greedy leaves anything on the table; §10 established only
that it leaves less than BLASSO does. It appears on the target §4's theory should fit best and
that §9.2 found best-behaved — which is suggestive and nothing more, at one instance per target.

**[A] The measured method is not quite §10's method, and this cuts against the finding.**
Removing continuous refinement is what isolates myopia, but it also removes the step that might
repair it: §10's greedy slides both atoms after placing them, and a refined pair starting from
greedy's grid choice could recover part or all of the 8.5%. So this bounds the myopia of
*grid* greedy, and transfers to §10's greedy only as an upper bound on what refinement has to
fix. Whether it does is untested and cheap to test — it needs only the same enumeration with a
polish step on both candidate pairs.

**[A] Two further limits.** $N=2$ on a discrete grid at $48^2$ is far from the regime of
interest, and whether myopia grows or washes out with $N$ is untested. Exhaustive best-3 is
$\binom{D}{3}\approx4\times10^{11}$ and out of reach; §11 U16 gives an affordable substitute.

### 10.4 Formulation, not solver — both sides solved exactly

`experiments/e4_exact_l0.py`, raw output `experiments/results/e4.txt`.

**The confound this removes.** Every comparison in §9 and §10 mixes two explanations for BLASSO
losing: the $\ell_1$ *formulation* is worse than $\ell_0$, or the *solver* failed. §9.3 shows it
demonstrably does fail past $N=16$, and §10's own caveat concedes "a better-engineered BLASSO
could plausibly close or reverse the gap" (U7). Nothing previously separated them.

**Method.** Shrink the problem until every route is solvable to optimality: a finite dictionary
on a $32^2$ image, $\ell_0$ by **exhaustive enumeration of all $\binom{D}{N}$ supports** (up to
75 million per row), $\ell_1$ by the **exact LARS path**, which is piecewise linear and therefore
exact at every breakpoint, then debiased on its own support. Greedy runs on the identical
dictionary. Whatever separates them is the formulation, because no solver failed.

**[V] Result, four targets, two dictionary panels, every row with $N\ge2$:**

- exact $\ell_1$ is **6–409% worse** than exact $\ell_0$ at matched $N$;
- greedy is **0–66% worse**, and beats $\ell_1$ in **every single row**;
- the $\ell_1$ support frequently shares **0 of $N$** atoms with the $\ell_0$-optimal support.

Representative rows (error as % of $\tfrac12\|y\|^2$, $D=768$):

| target | $N$ | $\ell_0$ exact | greedy | $\ell_1$ debiased | greedy excess | $\ell_1$ excess | shared support |
|---|---|---|---|---|---|---|---|
| cartoon | 2 | 18.00 | 23.11 | 30.90 | +28% | +72% | 0/2 |
| cartoon | 3 | 11.86 | 16.07 | 29.06 | +36% | +145% | 1/3 |
| ascent | 3 | 22.94 | 28.09 | 37.95 | +22% | +65% | 0/3 |
| face | 3 | 18.13 | 23.53 | 32.84 | +30% | +81% | 0/3 |
| in-model | 2 | 13.58 | 22.58 | 36.81 | +66% | +171% | 0/2 |

**[V] U7 is resolved in the negative for this regime.** §10's ordering survives when the solver
is removed as an explanation, and widens. **[A]** It does not follow that a better solver changes
nothing at $N=8$–$10^3$; it follows that solver quality is not what §10 was measuring.

**[A]** This is what §2.2 predicts. The relaxation cannot see $N$, so it selects a support
optimized for a different criterion, and debiasing on that support cannot repair a support chosen
wrongly in the first place.

**[V] It also calibrates §10.2's bound against a known optimum at $N>1$** for the first time. On
a finite dictionary $s(p)=\max_j|\langle g_j,p\rangle|$ is a maximum over $D$ numbers rather than
a search, so the bound is a theorem there. It still falls far below the true optimum — 11.01
against 33.18 at $N{=}1$ on cartoon, and 0.00 against 11.86 at $N{=}3$. **[A]** Confirming §10.2's
looseness is intrinsic and not a search artefact, exactly as §2.2 implies.

**[A] Scope.** $N\le4$ on a few hundred atoms, one instance per target, entirely on-grid.
Exhaustive enumeration is $\binom{D}{N}$, so this cannot be pushed to §10's $N=8$ or 16 — that
needs branch-and-bound (§2.2, §11 U19), which is not demonstrated here.

### 10.5 Dictionary design: a real effect that refinement mostly erases

`experiments/e4_exact_l0.py` (coherence sweep, designed dictionaries) and
`experiments/e5_dictionary_scaling.py`, raw output `results/e4_coherence.txt`,
`results/e4_design.txt`, `results/e5.txt`.

**[V] Coherence is extreme and drives the on-grid gaps.** The dictionary's maximum
$|\langle g_i,g_j\rangle|$ is 0.985 — the regime where every $\ell_1$ recovery condition fails.
Sweeping it down by target-independent random pruning collapses both gaps: on cartoon at $N{=}3$,
greedy's excess falls 35.5% → 0% and $\ell_1$'s 145% → 19% as coherence goes 0.985 → 0.75.
**[A] But it is not a free lunch** — the $\ell_0$ optimum itself worsens 11.86 → 22.20, so random
pruning trades away more approximation power than it buys.

**[V] A designed dictionary does better — but the original comparison was confounded.** Building
the dictionary the way §4's theorem does — parabolic scaling, orientations, a lattice adapted to
each atom's own axes — beat a dense unstructured sweep at half the size on all three targets at
$N{=}3$. **[A] That comparison was not fair.** A later audit found the two dictionaries did not
span the same scales, despite a code comment asserting they did: the unstructured sweep's major
axis topped out at **12px** against the parabolic one's **24px**. Large atoms carry an image's
smooth content cheaply, so part of the margin was simply bigger atoms.

**[V] Re-measured under a proper control** (`e5`-derived, three targets, budgets 8–64 splats):

| unstructured baseline | parabolic wins |
|---|---|
| original, major axis capped at 12px, $D=768$ | 12/12 — **confounded** |
| scale-matched but 4.6× larger, $D=1792$ | loses at 8 splats on all three targets |
| scale- **and** size-matched, $D=448$ vs 386 | **11/12, by 4–39%** |

**[A] The conclusion survives the control but the original framing did not earn it.** Structure
helps at matched size; dictionary *size* helps too, and the phrase "at half the size" was carrying
weight that belonged to the scale cap. **[A] This is still §4's first empirical contact** — U6
records the theorem as never tested.

**[V] Coherence is not the mechanism, though.** A difference-of-Gaussians dictionary has the
*lowest* coherence tested (0.881 at matched size) and performed **worst** of everything, because
zero-mean atoms cannot carry an image's mean. **[A]** Which is why `tip2006.pdf` pairs its
oscillatory sub-dictionary with a pure-Gaussian one — a detail §12 records without noting why it
matters.

**[V] And the advantage largely evaporates off-grid — this is the important row.** E5 budgets in
**splats** (a DoG atom is two splats, so E4's matched-atom comparison charged the mixed dictionary
half price), scales to 64 splats on $64^2$ images, and refines both sides identically:

| | on-grid | after identical off-grid refinement |
|---|---|---|
| structured vs unstructured | structured wins ~25%, nearly everywhere | shrinks to a few %, winner erratic |
| at 64 splats | mixed picture | **unstructured wins outright** — cartoon 0.778 vs 0.887, ascent 4.391 vs 4.734, face 2.367 vs 2.951 |

**[V] Refinement itself cuts error by 50–65%**, far more than any dictionary choice at any budget.

**[A] So the honest claim is narrow:** structure buys a better *starting point* at small budgets,
not a better answer. An earlier draft of this section claimed a 36–52% improvement from a mixed
dictionary; that was the splat-accounting error, and with correct accounting mixed and parabolic
are level and both wash out. Withdrawn — Appendix A.

### 10.6 Wavelet placement does not work

`experiments/e6_wavelet_init.py`, raw output `results/e6.txt`.

**Why it was worth testing.** §10.5 leaves one role where a transform could matter, and it is
**cost**, not quality. Every method here places atoms by searching the dictionary at every step,
and that search is what §9.3's ceiling and U5 are blocked on. A wavelet transform yields a
multiscale, oriented, position-resolved decomposition in $O(n^2)$ with no search. **[A]** This is
also what §4's theorem does read as an algorithm rather than a proof: threshold the curvelet
expansion, replace each retained curvelet by Gaussians. §4.1 dismisses it as non-adaptive, but
what is non-adaptive is the *budgeting* — *which* curvelets get approximated is chosen from the
data.

**[V] It loses, comprehensively.** Top DWT coefficients mapped to Gaussians, then refined
identically to §10.5, at matched splat budget, across `haar`, `bior2.2`, `db4`, `sym4`:

- the best wavelet loses to the best greedy in **12 of 12** cells, by **+1.7% to +69%**
  (median ≈ 31%). **[V]** The mapping's two free constants (`scale_k`, `aniso`) were guessed, so
  they were later swept: the chosen pair was best-of-grid on cartoon but *worst* of sixteen on
  ascent, which inflated that target's gap. Re-tuned **per target** — an advantage greedy is not
  given — the best wavelet still loses by 17–34%, so the verdict holds and only the magnitudes
  move;
- wavelet placement also loses to **random** placement in 10 of 12 cells;
- no family differs meaningfully from any other;
- placement is 8–215× cheaper, which does not compensate.

**[A] Mechanism.** A wavelet oscillates and one Gaussian cannot represent one wavelet
coefficient. **[V]** §4's own construction spends **4–11 Gaussians per retained curvelet** (its
budget sequence for $N=256$ begins 11, 8, 5, 5, 4, …), so a faithful transform-then-fit costs
several splats per coefficient. The 1:1 mapping tested here is the *optimistic* case and it
already loses.

**[V] Side finding, from a 5-seed control: greedy placement beats random placement in 12 of 12
cells by 10–45%.** Placement does matter. **[A]** An earlier single-draw reading suggested
otherwise and was wrong — Appendix A.

**[A] Scope.** Separable DWTs carry only three orientations per scale, so this is evidence
against *separable wavelets*, not against oriented systems generally — curvelets and shearlets
resolve orientation far better and are what §4 actually uses. Pre-registered before the run so a
null result is not over-read.

### 10.7 The frequency domain: one half is a no-op, the other does not help

`experiments/e7_frequency_continuation.py`, raw output `results/e7.txt`.

**[A] Fitting in the Fourier domain is provably empty.** By Parseval
$\|\Phi m-y\|^2 = n^{-2}\|\mathcal{F}(\Phi m)-\mathcal{F}(y)\|^2$, so the objective is identical
up to a constant — verified numerically to ten decimals. The Gaussian dictionary is moreover
closed under the transform ($\Sigma\mapsto\Sigma^{-1}$), so it maps to a dictionary of the same
family, and Parseval preserves every inner product, so the Gram matrix, the coherence, the local
minima and the global optimum are all literally unchanged. Re-deriving any of this in frequency
cannot alter the landscape. **[A]** Two things do change and both are already in play: the
transform as a *computational* device (`eta_sup` already correlates by FFT), and a frequency-
*weighted* $L^2$, which is a different objective but still Hilbertian, hence still certifiable —
§11 U21, and the only route to U9 that keeps §6.

**[A] Frequency continuation is a different algorithm and could have helped.** Low-pass the
target hard, fit, sharpen progressively, carry the solution forward. A blurred objective has
fewer local minima, so tracking the minimiser as the target sharpens is graduated non-convexity —
a classical route to a better optimum, and roughly what FreGS reports working for 3D splatting.
**[U]** That comparison is from search summaries, not the paper.

**[V] It does not help here.** Five-stage schedule ($\sigma=8,4,2,1,0$ px), three targets,
budgets 8–64 splats, greedy and random initialisation:

| allocation | result |
|---|---|
| equal total compute | continuation **loses in 22 of 24 cells**, by +1.8% to +27% |
| continuation handicapped — final stage gets the *full* budget, coarse stages free on top | roughly neutral: **better in 10 of 12 cells from random init** (median ≈ −1.8%), **worse in 9 of 12 from greedy init** (median ≈ +1.8%) |

**[A] The split is the finding.** Continuation is a way to escape a *bad* start: it helps random
initialisation consistently and hurts greedy initialisation, which is already well placed and
gets dragged off it by the coarse stages.

**[V] But the schedule above is nearly the worst one available, and that was not checked before
concluding** (`e10_schedule_sweep.py`, U22, three targets × two budgets × two initialisations,
medians over seeds, handicap allocation throughout):

| schedule $\sigma$ (px) | median vs direct | better | worse |
|---|---|---|---|
| $[1,0]$ | **−2.87%** | **9** | 3 |
| $[2,1,0]$ | −2.33% | 9 | 3 |
| $[4,2,1,0]$ | −1.40% | 7 | 5 |
| $[8,4,2,1,0]$ — *the one used above* | +0.82% | 4 | 8 |
| $[16,8,4,2,1,0]$ | +4.81% | 4 | 8 |

Control: the no-blur schedule $[0]$ reproduces direct refinement in 12/12 cells, so the
differences are the schedule and not the machinery.

**[V] The trend is monotone: the milder the schedule, the better continuation does**, and a mild
one genuinely helps — 2.9% median, better in 9 of 12 cells. **[A] So the verdict stated above is
schedule-dependent, and the schedule chosen for it was among the worst tested.** Corrected
reading: *at equal compute* continuation loses (22 of 24 cells, unchanged); *given free extra
compute for the coarse stages*, a mild two-stage schedule gives a small but consistent gain, and
the aggressive schedule originally tried gives none. **[A] It is still not a route to the global
optimum** — 2.9% is far from the 4–39% that dictionary choice moves, and the gain is bought with
compute that direct refinement was not given. But "frequency continuation does not help" was too
strong, and rested on one untuned parameter.

**[A]** The equal-compute row alone would have been misleading: splitting the budget across five
stages leaves the final stage — the only one run on the true target — a fifth of the iterations.
The handicap row exists to remove that confound, and it changes the verdict from "clearly harmful"
to "neutral".

**[V] A known-answer test sharpens this** (`verify_experiments.py`). On a target that is an exact
combination of dictionary atoms — optimum exactly 0 — started *at* that optimum and mildly
perturbed, direct refinement returns to $2.7\times10^{-5}$% every time. Continuation with a
trivial (no-blur) schedule reproduces direct exactly, confirming the machinery. Continuation with
a real schedule lands anywhere from $4\times10^{-4}$% to **27%** away, varying erratically with
the schedule *and* with the instance, in no stable pattern. **[A]** So continuation can lose a
solution it was handed, and direct refinement does not. That is a stronger statement than §10.7's
relative comparisons support on their own, and it is the mechanism behind the "worse from greedy
init" column. **[A] Not established:** an earlier reading attributed the loss specifically to the
coarsest ($\sigma=8$px) stage on the strength of one draw; a second draw reversed which schedule
failed, so only the erratic behaviour is supported, not a culprit stage. Whether a *tuned*
schedule would change §10.7's aggregate verdict is untested — §11 U22.

---

### 10.8 Branch-and-bound: the search works, the certificate does not

`experiments/e8_branch_and_bound.py`, raw output `results/e8_localsearch.txt`.

§2.2's positive half said exact $\ell_0$ branch-and-bound was the one route to a *certified*
global optimum it did not rule out, and §11 U19 made it the highest-value item. It is implemented
here. **[V] The certification half fails, for a structural reason that can be quantified.**

**Implementation.** Big-M formulation ($\|c\|_0\le N$, $\|c\|_\infty\le M$), best-first search so
the queue front is the global lower bound at every instant, node bound (\*) from §10.2
specialised to a node, and an incumbent from greedy plus exhaustive single-atom swaps. After
setup everything runs on the Gram matrix and $b=Gy$ — never on pixels — so a node costs
$O(D|I|)$.

**[V] The bound is vacuous, and not by a margin tuning can close.** Bound (\*) is non-trivial only
when $M\sigma/R<1$. Measured on cartoon, $D=248$, at the *tightest admissible* box — $M$ equal to
the incumbent's own largest amplitude, below which the incumbent itself becomes infeasible:

| | $N{=}4$ | $N{=}6$ |
|---|---|---|
| $M\sigma/R$ at $M=1.0\times\lvert c\rvert_{\max}$ | **5.7** | **8.9** |
| at $2.0\times$ | 11.4 | 17.8 |

Off by a factor of 6–9 where it needs to be under 1. 200 000 nodes returned a 100% gap.

**[A] Why, and it is §2.2 again.** The incumbent's largest amplitude is 24.0 against
$\|y\|=21.2$: a single atom's amplitude is the scale of the *whole signal*. So the mass budget
$NM\approx4\|y\|$, and a measure of that mass over a 248-atom dictionary fits $y$ easily. The
relaxation cannot see $N$ (§2.2), and when amplitudes sit at signal scale the mass constraint
does not bite either — so the only handle the convex hull leaves is one this problem does not
supply. **[A]** This is a property of the splatting dictionary, not of the implementation: no
choice of $M$, node ordering or search strategy repairs a relaxation that is loose by 6× before
the search starts.

**[V] The search half works, though.** Exhaustive enumeration of all **153 829 130** supports on
the parabolic dictionary at $N{=}4$ returns exactly the greedy-plus-swap solution (6.6895% both).
On the harder unstructured dictionary ($D=768$, coherence 0.985), adding swaps to greedy closes
most of §10.4's shortfall:

| target | $N$ | exhaustive | greedy | greedy + swap | still short |
|---|---|---|---|---|---|
| in-model | 2 | 13.58 | 22.58 | **13.58** | 0% |
| in-model | 3 | 0.00 | 10.47 | **0.00** | 0% |
| ascent | 3 | 22.94 | 28.09 | **22.94** | 0% |
| face | 2 | 32.09 | 35.77 | 32.45 | 1.1% |
| face | 3 | 18.13 | 23.53 | 19.77 | 9.0% |
| cartoon | 2 | 18.00 | 23.11 | 19.27 | 7.1% |
| cartoon | 3 | 11.86 | 16.07 | 15.32 | 29.3% |

**[V] Local search reaches the exact global optimum in 4 of 8 cells** and cuts greedy's shortfall
from 0–66% to 0–29% in the rest. **[A]** So at these budgets the optimum is often *reachable*
cheaply; what is unavailable is a *proof* that it was reached. That is the honest state of the
central question: we can frequently find the global optimum, and cannot certify it by any route
tested here.

**[A] Scope.** $N\le6$, $D\le768$, on-grid, one instance per target. A negative certification
result for *this* bound on *this* dictionary — the perspective/ridge relaxations used by the
sparse-regression literature are stronger and untested here (§11 U19 restated).

### 10.9 The perspective relaxation closes U19, negatively

`experiments/e9_perspective.py`, raw output `results/e9.txt`.

§10.8 blocked on the big-M bound and left one candidate: the **perspective
relaxation**, the tool the sparse-regression literature actually uses. This settles it by
evaluating the relaxation at the **root** — a relaxation weak before the first branch cannot be
rescued by any search strategy, so a large root gap is decisive against, while a small one would
have justified building a second solver.

**[A] Derived, not cited.** All three sources (`irit.fr`, `optimization-online`, the arXiv
mirrors) are blocked by the egress proxy, so nothing here rests on a search summary. Writing the
cardinality-constrained ridge problem as a mixed-integer program and replacing $c_j^2$ by its
perspective $c_j^2/z_j$, the inner minimization over $z$ is a water-filling:

$$\Omega(c)=\min\Big\{\textstyle\sum_j c_j^2/z_j \;:\; \sum_j z_j\le N,\ 0\le z_j\le1\Big\} = \sum_{\rm sat} c_j^2 + \Big(\sum_{\rm rest}|c_j|\Big)^2\!\Big/(N-s)$$

**[V] Verified numerically:** $\Omega(c)=\|c\|^2$ *exactly* when $c$ has at most $N$ non-zeros,
and strictly exceeds it otherwise. So $\min_c \tfrac12\|y-Gc\|^2+\lambda_2\Omega(c)$ is a genuine
lower bound on the $N$-sparse ridge optimum, tight on every feasible point.

**[A] The ridge is not optional.** As $\lambda_2\to0$ the penalty vanishes and the bound
degenerates to an unconstrained least-squares fit over the whole dictionary. So this relaxation
can only certify a problem that *carries* a ridge, and what it certifies is $\ell_0$+ridge, not
(P0). The question is therefore not whether it works but **how much ridge it needs, and whether
that much ridge still describes the problem**.

**[V] Both measured, against the exhaustively computed $N$-sparse ridge optimum** ($D=248$,
$N=3$; `l2 err` is the pure reconstruction error, comparable with every other table here):

| $\lambda_2$ | root gap, cartoon | root gap, ascent | ridge/data | `l2 err` cartoon | `l2 err` ascent |
|---|---|---|---|---|---|
| 0 | 85.5% | 79.5% | 0.00 | 7.69% | 19.48% |
| $10^{-3}$ | 73.9% | 64.2% | 0.03 | 7.69% | 19.49% |
| $10^{-2}$ | 51.6% | 46.3% | 0.30 | 7.77% | 20.15% |
| $10^{-1}$ | 15.2% | 13.6% | 0.52 | **13.48%** | 24.55% |
| $1$ | **0.33%** | **0.04%** | 0.65 | **33.34%** | **47.67%** |
| $10$ | 0.00% | 0.00% | 0.10 | 83.02% | 87.73% |

**[V] There is no window in which the relaxation is tight and the problem is intact.** Where the
ridge is negligible ($\lambda_2\le10^{-3}$, reconstruction error unchanged to four decimals) the
gap is 64–86%. Where the gap closes ($\lambda_2=1$, gap 0.04–0.33% — branch-and-bound would
terminate at the root) the reconstruction error has risen from 7.7% to 33.3% and from 19.5% to
47.7%. Buying a certificate costs a 2.4–4.3× worse encoding, and the two requirements move in
opposite directions monotonically across four orders of magnitude.

**[A] So U19 closes negatively, and for a reason specific to this problem.** §10.8 diagnosed it:
a single atom's amplitude is the scale of $\|y\|$ itself, so no mass-, norm- or ridge-based
relaxation has anything to bite on until the regularizer is strong enough to dominate the data
term. Both the big-M and the perspective route fail the same way, which is evidence the
obstruction is the splatting dictionary rather than the choice of relaxation.

**[A] Scope, pre-registered.** A root gap is necessary, not sufficient — a *small* root gap would
not have proven the tree small, and a large one is strong evidence rather than proof. One
dictionary, two targets, $N=3$, on-grid.

**[A] The instrument caught itself, again.** The first run reported *negative* root gaps — a
relaxation exceeding the quantity it relaxes, which is impossible. $\Omega$ is convex but
**nonsmooth**: for an unsaturated coordinate $z_j=|c_j|/\sqrt{\mu}$, so
$c_j/z_j=\sqrt{\mu}\,\mathrm{sign}(c_j)$ and the gradient jumps at zero exactly as $\ell_1$ does,
stalling L-BFGS above the true minimum. Repaired by smoothing the magnitudes and warm-starting at
the exact optimum, both of which can only move the value *downward*; check **C5** now asserts
relaxation $\le$ optimum and passes on every row. **[A]** Note the direction: the bug made gaps
look *smaller*, so the negative conclusion was never at risk — but the $\lambda_2\ge1$ rows, which
are the ones that decide the trade-off, were entirely wrong before the fix.

## 11. Open questions, and what would settle each

| | question | what would settle it | cost |
|---|---|---|---|
| **U1** | Does GaussianImage's implementation clamp or otherwise break linearity (§3.1)? | Read the released CUDA kernel and loss; check for output clamping or an activation on $c'_n$ | hours |
| **U2** | Does the certificate generalize to colour (§3.3)? | Derive the group-BLASSO dual, verify $\|\eta(\theta)\|_2\le1$; obtain a primary source on vector-valued BLASSO | days |
| **U3** | Is A1's boundarylessness essential or technical (§7.1)? | Check whether CPGD's proof localizes when the optimum's scales are interior; failing that, run CPGD on truncated $\Theta$ and test against the predicted linear rate | weeks |
| **U4** | Is the recovery/approximation mechanism behind the hump real (§9.1)? | Hold $K_{BL}$ exactly at $K$ with a support-constrained solve and re-sweep separation; if the hump persists, $\lambda$-matching and count drift are excluded | days |
| **U5** | Can the solver certify at realistic $N$ (§9.3)? | Replace add/prune with batched addition and a support-aware prune rule, or an exact LASSO inner solve; target certgap $<0.1\%$ at $N=10^3$ | weeks — **gates U7, U8** |
| **U6** | Does §4's rate describe real solutions? | Fit a cartoon target at several $N$; regress $\log$(minor axis) on $\log$(major axis) for edge atoms — parabolic scaling predicts slope 2. Descriptive only; absence is not evidence of failure (§4.1) | days |
| **U7** | Does §10's negative result survive scale and a better solver? | **Half answered — §10.4.** With both sides solved *exactly* at $N\le4$, $\ell_1$ loses to $\ell_0$ by 6–409%, so solver quality is not what §10 measured. Whether it survives *scale* is still open: re-run §10 after U5 at $N\ge10^3$ on $\ge256^2$ images, ≥5 instances per cell with error bars | weeks, after U5 |
| ~~U19~~ | ~~Can exact $\ell_0$ branch-and-bound certify at §10's budgets?~~ | **Resolved, negatively — §10.8, §10.9.** Both standard node relaxations fail on this dictionary. The big-M bound is loose by 6–9× at the tightest admissible box; the perspective relaxation is loose by 64–86% wherever the ridge is small enough to leave the problem intact, and only closes at a ridge that makes the encoding 2.4–4.3× worse. The cause is shared: a single atom's amplitude is the scale of $\|y\|$, so no norm-based relaxation binds. **This was the last route §2.2 left open to a certified global optimum** | done |
| **U20** | Does §10.5's dictionary effect survive real scale? It is measured at $\le64$ splats on $64^2$, and it already inverts at 64 | Re-run §10.5 at $10^3$ splats on $\ge256^2$; needs no solver work, so unlike U7 it does **not** wait on U5 | days |
| **U23** | Does canonicity survive off-grid and at larger $N$ (§10.10)? At $N{=}3$ on a grid the optimum is unique and stable; at 32 splats off-grid no method reaches the same answer twice. The grid's minimum separation may be what creates the uniqueness | Enumerate at $N{=}5$–6 on a reduced dictionary to test the $N$ direction; for the grid direction, run many random+Adam restarts from near the known on-grid optimum and check whether they return to it or disperse | days — **now the most important open question** |
| ~~U22~~ | ~~Does §10.7's verdict on frequency continuation depend on the blur schedule?~~ | **Resolved — yes, §10.7.** The schedule used was among the worst of five swept. A mild $[1,0]$ schedule gives −2.87% median, better in 9/12 cells, against +0.82% for the one originally used; the trend is monotone in schedule aggressiveness. The gain is small and requires the handicap allocation, so continuation is still not a route to the optimum, but the original verdict was too strong | done |
| ~~U22-old~~ | ~~superseded~~ A known-answer test shows continuation losing a handed-in optimum by 4e-4% to 27%, erratically across schedules and instances, so the single schedule §10.7 used may not be representative | Re-run §10.7 sweeping the schedule (number of stages, coarsest $\sigma$), medians over $\ge5$ seeds since single draws demonstrably reverse | days |
| **U21** | Does a frequency-weighted $L^2$ buy perceptual quality while keeping the certificate (§3.2, U9)? A weighted $L^2$ is still Hilbertian, so the adjoint and §6 survive, which SSIM and $L^1$ do not | Fit under a contrast-sensitivity weighting, compare against plain $L^2$ on a perceptual metric at equal $N$ | days; needs the U9 metric decision |
| **U8** | Do any §9 conclusions survive $10^3$–$10^5$ atoms? | Re-run §9 after U5 | after U5 |
| **U9** | Is $L^2$-optimal perceptually acceptable (§3.2)? | Compare $L^2$-optimal against SSIM-trained fits at equal $N$, human or perceptual metric | days; needs a metric decision |
| **U10** | Does matching a Zador-type density actually reduce reconstruction error (§7.2)? | Place $N$ atoms by semi-discrete OT at several candidate density laws; compare resulting $L^2$ error against matched-filter placement at equal $N$. If no law wins, the OT guarantee is rigorous about an irrelevant objective | days |
| **U11** | Is the §9.1 hump present out-of-model? | §9.2 sweeps budget, not separation, so the two are not comparable. Sweep $r_{\rm ref}$ on a fixed out-of-model target by varying $N$ and image scale together, and check for a peak | days |
| **U12** | How much of the out-of-model penalty is $\ell_1$ bias vs. no exact representation? | Repeat §9.2 on targets that *are* exact mixtures but at matched $r_{\rm ref}$ and $N$; the difference isolates the model-class effect | days |
| **U15** | Is §10.2's decay in $N$ real, or is it slack? The bound covers encodings of mass $\le M_g$ with *any* atom count, and is already vacuous at $1.5M_g$ | **No clean route is known** — cardinality does not dualize, so the relaxation cannot simply be tightened away. What is affordable: re-run greedy under an explicit mass cap so method and bound are matched at the same $M$, and check whether greedy's error rises to meet the bound (decay is real) or does not (decay is slack) | days |
| **U16** | Does greedy's $N{=}2$ myopia (§10.3) grow or wash out with $N$? | Exhaustive best-3 is $\approx4\times10^{11}$ subsets, so instead sweep the first atom over the top-$K$ grid candidates, run full continuous greedy from each, and compare against standard greedy at $N=8,16$ | days |
| **U17** | Does the boundary norm defect (§7.1.2) change any result? | Renormalize the dictionary by the *measured* discrete norm rather than the analytic one, re-run §9.1 and §10. §10.2 found no cost at $N{=}1$; this tests everywhere else | days |
| **U18** | Are §9 and §10 reproducible from the repository? Their committed defaults do not match the reported tables and no raw output is stored (§9 preamble) | Re-run E2 and E2b at the parameters the tables actually used, commit the output alongside as E3 does, and reconcile any row that moves | hours — **do before U7/U8, which re-run both** |
| ~~U13~~ | ~~Which uncertified method beat BLASSO — greedy or restarts?~~ | **Resolved — §10.1.** Greedy, in all nine rows, matching $E_{\rm ref}$ exactly | done |
| ~~U14~~ | ~~How far is *greedy* from the (P0) optimum out-of-model?~~ | **Partly resolved — §10.2.** Greedy is *exactly* optimal at $N{=}1$ on all four targets; the bound itself certifies only 24–34% there and goes vacuous by $N=4$–8. **Still open at $N\ge8$**, i.e. at §10's own budgets, and the residue is U15. The method originally proposed here was wrong twice over: an exhaustive grid with least-squares amplitudes returns a *feasible point*, hence an **upper** bound on $E_{\rm opt}$, which cannot bound greedy's distance from it; and $\binom{10^5}{8}\approx10^{36}$ is not enumerable. See Appendix A, M8 | partly done |

**[A] Dependency structure.** U5 gates U7 and U8, and those two decide whether this line of work
has a future. U1 is cheap and should be first, since §3.1 is a premise for everything. U4, U6,
U10, U15, U16 and U17 are cheap and independent.

**[A] U19 is now the highest-value item.** §2.2 rules out measure-space convexification as a
route to (P0)'s optimum, and §10.4 confirms the cost empirically with solvers removed as an
excuse. What §2.2 does *not* rule out is the discrete route: on a finite dictionary with bounded
amplitudes the convex hull does depend on $N$, which is what exact branch-and-bound uses. That is
the only remaining path to a *certified global* optimum at the budgets §10 actually reports, and
nobody here has tried it.

**[A] U15 and U16 remain what most limit the bound-based route.** §10.2 supplies an absolute
bound but loses it exactly where §10 makes its case, and §10.3 shows greedy's placement is not
optimal even where the optimum is computable.

**[A] Be precise about how little §10.2 excludes.** The reading "greedy is the best of two
mediocre options" is ruled out only at $N=1$, where greedy is exactly optimal. At $N=2$–6 the
bound permits suboptimality of 42–98%, which excludes almost nothing, and at $N\ge8$ it permits
everything. One instance per target, four targets.

**[A]** U15 is the only open question that could yield an *absolute* statement at realistic $N$
without first solving U5 — several others (U1, U6, U10, U16, U17, U18) are also independent of
U5, but none of them bounds the optimum.

---

## 12. Programme

**P0 — U18.** Make §9 and §10 reproducible before anything re-runs them. Hours, and every later
step depends on the numbers being recoverable.

**P1 — U1.** Confirm the forward model. Cheapest check on the load-bearing premise.

**P2 — ~~U19~~, closed.** Exact $\ell_0$ branch-and-bound was the last route §2.2 left open to a
*certified* global optimum. §10.8 and §10.9 close it: both standard node relaxations fail on this
dictionary, for the same reason, and the reason is a property of the splatting atoms rather than
of the method. **[A] There is now no known route to a certificate here.** What remains is to
*find* good solutions and measure them against exhaustive enumeration wherever that is affordable
— which §10.8 shows greedy-plus-swap already does well.

**P2b — U5.** Fix the add/prune loop until it certifies at $N\ge10^3$. Still worth doing — it is
what makes §6 usable as the diagnostic instrument §10 demotes it to — but §2.2 and §10.4 show it
leads to the global optimum of the wrong problem, so it is not the thing standing between this
work and an optimal encoding.

**P2c — the honest replacement for both.** Since certification is unavailable, the tractable
question is how good the *reachable* solutions are. §10.8's greedy-plus-swap attains the
exhaustively verified optimum in half the cells tested; extending that local search (larger
neighbourhoods, multiple restarts) and calibrating it against enumeration at every $N$ where
enumeration is affordable is cheap, needs no solver work, and is the only thing here that
produces numbers about (P0) itself.

**P3 — U7/U8.** Re-run §10 and §9 at scale with error bars. If greedy still wins, the honest
conclusion is that the convex route is a diagnostic tool and matching pursuit is the *better*
encoder — which §10.2 and §10.3 show is not the same as the *good* one.

**P4 — U4, U6, U15, U16.** Cheap, independent, and they test the interpretive claims currently
resting on hypothesis. **[A]** U15 deserves priority among them: it is the only route to an
absolute statement at realistic $N$ that does not wait on U5, because §10.2's bound is valid
whether or not any solver converged.

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

Rules adopted after auditing this document's own reversals. Thirteen substantive claims were
stated here and later withdrawn. Two further errors (M8's second half, M9) were caught before
reaching the document and are logged anyway, since neither would have been caught by prose
review.

The twelfth and thirteenth are recent and both ran in the same direction — overstating a new
result. A mixed Gaussian/DoG dictionary was reported as improving on the unstructured one by
36–52%; the comparison was budgeted in dictionary *atoms*, but a DoG atom is two splats, so the
mixed dictionary was charged half price (§10.5). And random placement was reported as competitive
with greedy on the strength of a *single* random draw; five draws reverse it in 12 of 12 cells
(§10.6).

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
**M8 — State which side a bound falls on, and check it before proposing it.** *Cost:* U14 stood
as this document's most limiting open question, with a proposed method — exhaustive grid search
— that returns a feasible point and therefore bounds the optimum from *above*. It could never
have answered the question it was written for, and the error survived several audits because
"exhaustive search" reads as authoritative regardless of direction. Corrected in §10.2.
**M10 — Budget in the unit the system actually pays.** A comparison is only matched if it is
matched in the resource being spent. *Cost:* §10.5's dictionary result was stated at 36–52%
improvement and is worth roughly nothing once the comparison is budgeted in splats rather than in
dictionary atoms. The error was invisible in the table, because both columns said "N".
**M11 — Compare a deterministic method against a distribution, not a draw.** *Cost:* a single
random initialisation appeared to beat greedy placement; the median of five loses to it in every
cell. Any control with a random seed needs several.
**M9 — Attribute a spread by conditioning, not by eyeballing a summary statistic.** *Cost:* the
§7.1.2 norm defect was first attributed to shear on the strength of two quartile medians, which
can only speak to the bulk and not the tail. Conditioning gave $\rho=-0.082$ for shear against
$+0.684$ for edge distance — the opposite conclusion. Caught before it entered this document,
and logged because the same reasoning would not have been caught in prose.

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
