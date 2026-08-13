# Convexification cannot see the atom count

A self-contained account of one result and its consequences: **over measures, the tightest convex
relaxation of "at most $N$ atoms" forgets $N$.** Everything else in this document either proves
that, bounds it, or measures what it costs.

Companion to `optimal-encoding.md`, which contains the experiments cited here. This document
repeats their numbers but not their methodology.

**Status tags,** as in the main document. **[V]** verified — proved here, or measured by an
experiment in `experiments/` · **[A]** analysis from **[V]** facts · **[U]** unverified, search
summaries only.

---

## 1. What is claimed, in one paragraph

Fix a parameter space $\Theta$ of Gaussian atoms and consider encodings $\mu=\sum_i a_i
\delta_{\theta_i}$ as measures on $\Theta$. The set of encodings using at most $N$ atoms is not
convex — average a one-atom encoding at $\theta_a$ with one at $\theta_b$ and you get two atoms —
so any convex method optimizes over its convex hull instead. **That hull is a ball in
total-variation norm, and the only thing it remembers about $N$ is the product of $N$ with the
per-atom amplitude bound.** Two atoms of mass $M$ are indistinguishable, after convexification,
from one atom of mass $2M$ — because two atoms can be moved together until they *are* one. The
result is sharp: it holds with or without a per-atom amplitude cap, and it fails on a finite
dictionary. The dividing line is whether two atoms can be moved arbitrarily close together.

Three things follow. The relaxation gap between $\ell_0$ and $\ell_1$ over measures is structural
rather than a defect of the $\ell_1$ penalty. A dual certificate for the convex problem cannot
certify optimality at fixed $N$. And the discrete route — a finite dictionary — is not a
computational convenience but the thing that makes sparsity convex-representable at all.

**[A] Which parts are textbook.** That the extreme points of the total-variation ball are the
signed Diracs, and hence that the ball is their closed convex hull, is standard functional analysis
(Lemma 2 below proves the part that is used, so nothing is imported on trust). What is not standard
is the behaviour of the **per-atom amplitude cap**: capping every atom at $M$ defines a strictly
smaller convex set before closure, and the cap is destroyed by atoms colliding (Theorem 1). That
cap is the big-$M$ constraint every mixed-integer sparse solver relies on, and its collapse is what
distinguishes the continuum from a grid. §3 and §4 are derived here; the literature reading behind
them is recorded in §10.

---

## 2. Setting

$\Theta\subset\mathbb{R}^p$ is the atom parameter space: centre, log-scales, orientation or shear.
Take it **compact** — centres in the closed image square, log-scales in a closed interval bounded
away from $\pm\infty$. Bounding $\sigma$ away from $0$ is what makes $\Theta$ compact in the
log-scale parametrization, and it is also what keeps $\varphi$ continuous: under pointwise sampling
$\varphi_\theta(x)\to\mathbf{1}\{x=\text{centre}\}$ as $\sigma\to0$, a limit that jumps according to
whether the centre lands exactly on a pixel, so $\theta\mapsto\varphi_\theta$ admits no continuous
extension to $\sigma=0$.

The image has finitely many pixels, $P$ of them, so $\varphi:\Theta\to\mathbb{R}^P$,
$\theta\mapsto\varphi_\theta$, and we assume it continuous — true for Gaussians with scales bounded
away from zero.

$\mathcal{M}(\Theta)$ is the space of finite signed Radon measures on $\Theta$, which is
$C(\Theta)^*$; it carries the weak-\* topology, in which $\mu_k\to\mu$ means $\int f\,d\mu_k\to\int
f\,d\mu$ for every continuous $f$. Write

$$\Phi\mu=\int_\Theta \varphi_\theta\,d\mu(\theta)\in\mathbb{R}^P,\qquad
J(\mu)=\tfrac12\|\Phi\mu-y\|^2,\qquad
B_\rho=\{\mu:|\mu|(\Theta)\le\rho\}$$

where $|\mu|(\Theta)$ is the total-variation norm, equal to $\sum_i|a_i|$ for
$\mu=\sum_i a_i\delta_{\theta_i}$.

**[A] $\Phi$ is weak-\*-to-norm continuous**, and this is where the finite pixel count earns its
keep. Each coordinate $\mu\mapsto\int\varphi_\theta(x)\,d\mu(\theta)$ is weak-\* continuous by
definition, since $\varphi_\cdot(x)\in C(\Theta)$; with finitely many coordinates, coordinatewise
convergence is norm convergence. Hence **$J$ is weak-\* continuous**. (For a continuum of pixels
one would need a uniform-integrability argument; nothing here depends on that case.)

The two families of interest, for $N\ge1$ and $M>0$:

$$\mathcal{F}_{N,M}=\Big\{\sum_{i=1}^{N}a_i\delta_{\theta_i}\;:\;\theta_i\in\Theta,\ |a_i|\le M\Big\},
\qquad
\mathcal{F}_N^{\,\tau}=\Big\{\sum_{i=1}^{N}a_i\delta_{\theta_i}\;:\;\theta_i\in\Theta,\
\textstyle\sum_i|a_i|\le\tau\Big\}$$

Both consist of sums of at most $N$ Diracs — taking some $a_i=0$ allows fewer — and neither admits
a diffuse part, so they are the $N$-atom encodings in the sense (P0) means.

$\mathcal{F}_{N,M}$ is the $N$-sparse family with a **per-atom** cap — the measure-space form of
the big-$M$ constraint that mixed-integer solvers use. $\mathcal{F}_N^{\,\tau}$ caps only the
**total** mass.

---

## 3. The theorem

> **Theorem 1 (collision).** Let $\theta^\*$ be an accumulation point of $\Theta$ and $N\ge2$.
> Then $NM\,\delta_{\theta^\*}$ lies in the weak-\* closure of $\mathcal{F}_{N,M}$.

*Proof.* Since $\theta^\*$ is an accumulation point, choose for each $k$ a set of $N$ distinct
points $\theta_1^k,\dots,\theta_N^k\in\Theta$ with $\theta_i^k\to\theta^\*$ as $k\to\infty$. Put
$\mu_k=\sum_{i=1}^N M\delta_{\theta_i^k}\in\mathcal{F}_{N,M}$. For $f\in C(\Theta)$,
$\int f\,d\mu_k=M\sum_i f(\theta_i^k)\to NMf(\theta^\*)$ by continuity. $\square$

The cap said *no atom may carry more than $M$*. The limit carries $NM$ at a single point. **The cap
does not survive closure.**

> **Lemma 2 (Diracs generate the ball).** For any $\rho>0$,
> $\overline{\operatorname{conv}}\,\{\pm\rho\,\delta_\theta:\theta\in\Theta\}=B_\rho$, closure in
> the weak-\* topology.

*Proof.* ($\subseteq$) $B_\rho$ is convex and weak-\* closed and contains each $\pm\rho\delta_\theta$.

($\supseteq$) Let $|\mu|(\Theta)\le\rho$. Fix $m$ and partition $\Theta$ into Borel sets
$B_1,\dots,B_{k_m}$ of diameter $<1/m$; pick $\theta_j\in B_j$ and set $a_j=\mu(B_j)$,
$\mu_m=\sum_j a_j\delta_{\theta_j}$. For $f\in C(\Theta)$, uniform continuity gives
$|\int f\,d\mu_m-\int f\,d\mu|\le\sum_j\int_{B_j}|f(\theta_j)-f(\theta)|\,d|\mu|\le
\omega_f(1/m)\,\rho\to0$, so $\mu_m\to\mu$ weak-\*. And $\sum_j|a_j|\le|\mu|(\Theta)\le\rho$, so
with $\lambda_j=|a_j|/\rho$ and $\lambda_0=1-\sum_j\lambda_j\ge0$,

$$\mu_m=\sum_j \lambda_j\big(\rho\,\mathrm{sign}(a_j)\,\delta_{\theta_j}\big)+\lambda_0\cdot 0,
\qquad 0=\tfrac12(\rho\delta_{\theta_1})+\tfrac12(-\rho\delta_{\theta_1})$$

exhibits $\mu_m$ as a finite convex combination of points $\pm\rho\delta_\theta$. Hence
$\mu\in\overline{\operatorname{conv}}\{\pm\rho\delta_\theta\}$. $\square$

**[A]** Lemma 2 is Krein–Milman applied to the weak-\* compact convex set $B_\rho$, whose extreme
points are exactly $\{\pm\rho\delta_\theta\}$; the proof above avoids invoking it, so nothing here
rests on a theorem not verified in place.

> **Theorem 3 (annihilation).** Let $\Theta$ be compact and **perfect** — every point of $\Theta$
> is an accumulation point of $\Theta$. Then for every $N\ge1$ and $M>0$,
> $$\overline{\operatorname{conv}}\,\mathcal{F}_{N,M}=B_{NM},
> \qquad\text{and}\qquad \overline{\operatorname{conv}}\,\mathcal{F}_N^{\,\tau}=B_\tau .$$

*Proof.* $\mathcal{F}_{N,M}\subseteq B_{NM}$ and $B_{NM}$ is convex and weak-\* closed, giving
$\subseteq$. For $\supseteq$, fix any $\theta\in\Theta$. If $N=1$ then $\pm M\delta_\theta\in
\mathcal{F}_{1,M}$ directly. If $N\ge2$ then $\theta$ is an accumulation point by hypothesis, so
Theorem 1 places $NM\delta_\theta$ in the closure of $\mathcal{F}_{N,M}$, and $-NM\delta_\theta$
likewise on taking $a_i=-M$. Either way $\pm NM\delta_\theta\in\overline{\mathcal{F}_{N,M}}$ for
every $\theta\in\Theta$, so Lemma 2 with $\rho=NM$ gives
$\overline{\operatorname{conv}}\,\mathcal{F}_{N,M}\supseteq B_{NM}$. The second identity is the
case $M=\tau,N=1$ together with
$\mathcal{F}_1^{\,\tau}\subseteq\mathcal{F}_N^{\,\tau}\subseteq B_\tau$. $\square$

**[A] The hypothesis is satisfied here and is not decorative.** The Gaussian parameter space is a
product of non-degenerate closed intervals — centre coordinates, log-scales, orientation — which is
compact and perfect, so every point is an accumulation point and Theorem 3 applies as stated. It is
also the hypothesis that fails for a finite dictionary, which is the subject of §4. A space mixing
isolated and non-isolated points falls between the two theorems and is not treated here; no such
space arises in this work.

**[A] What the theorem says in words.** After convexification, the $N$-sparse family with per-atom
cap $M$ is *exactly* the total-variation ball of radius $NM$. $N$ and $M$ enter only through their
product. Halving the atom budget and doubling the allowed amplitude gives literally the same convex
set.

> **Corollary 4 (no convex penalty separates).** Let $R:\mathcal{M}(\Theta)\to(-\infty,\infty]$ be
> convex and weak-\* lower semicontinuous. If $R\le c$ on $\mathcal{F}_{N,M}$ then $R\le c$ on all
> of $B_{NM}$.

*Proof.* $\{R\le c\}$ is convex and weak-\* closed, so it contains
$\overline{\operatorname{conv}}\,\mathcal{F}_{N,M}=B_{NM}$. $\square$

So no convex regularizer whatever — not $\ell_1$, not a reweighted, spatially varying, or
group-structured variant — has a sublevel set that admits the $N$-sparse encodings and rejects any
measure of total mass at most $NM$, however many atoms it has or whether it has any at all. This is the precise sense in which the atom count is annihilated.

**[A] The lower-semicontinuity hypothesis is doing real work and cannot simply be dropped**, since
a convex function with non-closed sublevel sets is not covered. It is not a loophole worth chasing:
lower semicontinuity is what makes minima exist and what any convergent scheme delivers in the
limit, and Corollary 5 below reaches the same conclusion *at the level of optimal values* with no
continuity or closedness assumption on the relaxation at all.

> **Corollary 5 (values, not just sets).** $\displaystyle\inf_{\mu\in\operatorname{conv}
> \mathcal{F}_{N,M}}J(\mu)=\inf_{\mu\in B_{NM}}J(\mu)$.

*Proof.* $J$ is weak-\* continuous (§2), so its infimum over a set equals its infimum over the
closure of that set; apply Theorem 3. $\square$

**[A] Corollary 5 closes the escape, and it is the sharpest statement here.** The *plain* convex
hull really is smaller than the ball: every element of $\operatorname{conv}\mathcal{F}_{N,M}$ is a
finite convex combination $\sum_j\lambda_j\mu_j$, whose mass at any single point is
$\sum_j\lambda_j\mu_j(\{\theta\})\le M$, so the cap does survive before closure. It buys nothing,
because the *infimum* over that smaller un-closed set is already the infimum over the whole ball.
The collision limit is not an artefact of insisting on closed sets; it is reached by minimizing
sequences.

**[A] And it says what happens as the cap is loosened, which is what one actually observes.**
(P0) itself has no amplitude cap; capped-(P0) becomes it once $M$ exceeds the largest amplitude
the optimum uses. But by Corollary 5 the relaxation's value is that of the ball of radius $NM$,
which grows without bound as $M$ does, so the relaxation degenerates to unconstrained least squares
over the whole dictionary — a bound of zero, true and useless. The cap cannot be loosened toward
the problem one wants without the relaxation collapsing, and it cannot be tightened past the
incumbent's own largest amplitude without cutting off the optimum. That squeeze is not a numerical
accident: §7 records it happening on both routes tried, the big-$M$ box and the perspective
reformulation, whose $\lambda_2\to0$ limit is exactly this degeneracy.

**[A] This also disposes of the topology objection**, which is the first thing a careful reader
should raise. The closure in Theorem 3 is weak-\*, and the choice matters: in the **total-variation
norm** topology the cap *would* survive, since $\mu_k\to\mu$ in TV implies
$\mu_k(\{\theta\})\to\mu(\{\theta\})$ pointwise, so a TV-closed convex hull of
$\mathcal{F}_{N,M}$ is strictly smaller than $B_{NM}$. One might therefore hope to build a
relaxation in the TV topology and keep $N$. Corollary 5 forecloses it: the collapse it states is a
statement about the **un-closed** hull, in which no topology appears at all. Whatever topology one
prefers, the best value any convex relaxation of the $N$-sparse capped problem can return is the
value of the plain mass-ball problem. (The weak-\* topology is also the natural one on other
grounds — bounded sets are weak-\* compact by Banach–Alaoglu and are not TV-compact, so minimizing
sequences have weak-\* limits and generally no TV limit.)

---

## 4. Where it fails: finite dictionaries

> **Theorem 6 (discrete).** Let $1\le N<D$ be an integer and $M>0$. Then in $\mathbb{R}^D$
> $$\operatorname{conv}\{c:\|c\|_0\le N,\ \|c\|_\infty\le M\}
> =\{c:\|c\|_1\le NM\}\cap\{c:\|c\|_\infty\le M\}=:P_{N,M}$$
> and for $N\ge2$ this is a **strict** subset of $\{c:\|c\|_1\le NM\}$.

*Proof.* ($\subseteq$) Every $N$-sparse $c$ with $\|c\|_\infty\le M$ satisfies $\|c\|_1\le NM$; both
constraints are convex.

($\supseteq$) $P_{N,M}$ is a compact polytope, hence the convex hull of its vertices, so it suffices
that every vertex is $N$-sparse with entries $\pm M$. Let $c$ be a vertex and let $k=\#\{j:|c_j|=M\}$.
If the constraint $\|c\|_1\le NM$ is inactive, the only active constraints are of the form
$|c_j|=M$, and a vertex needs $D$ active independent constraints, forcing $|c_j|=M$ for all $j$ and
so $\|c\|_1=DM>NM$ — a contradiction for $N<D$. So $\|c\|_1=NM$, and the coordinates with
$0<|c_j|<M$ contribute $(N-k)M$ in total. If two such coordinates were nonzero, moving mass
between them would keep both $\|c\|_1$ and $\|c\|_\infty$ fixed and exhibit $c$ as a midpoint,
contradicting extremality; so at most one such coordinate exists. If one does, call it $c_r$, then
$|c_r|=(N-k)M$, and $0<|c_r|<M$ forces $0<N-k<1$, impossible for integers. So no coordinate lies
strictly between, giving $\|c\|_1=kM=NM$ and hence $k=N$: exactly $N$ coordinates equal $\pm M$ and
the rest vanish.

(Strictness) For $N\ge2$ the vector $NMe_1$ has $\|c\|_1=NM$ but $\|c\|_\infty=NM>M$. $\square$

**[A] The contrast is exactly the collision.** In $\mathbb{R}^D$ the atoms sit at fixed, distinct
coordinates and cannot be moved together; the cap $\|c\|_\infty\le M$ therefore survives, and the
resulting convex set depends on $N$ and $M$ separately. In $\mathcal{M}(\Theta)$ two atoms may be
brought arbitrarily close, their masses add in the limit, and only the product survives.

**[A] The two clean cases are topological, and a compact space in which every point is isolated is
finite.** So a compact parameter space with no isolated points falls under Theorem 3 — the cap is
annihilated for every $N\ge2$ — and a compact one with only isolated points is finite and falls
under Theorem 6, where the cap survives. Spaces mixing the two interpolate and are not treated.
**Whether sparsity is convex-representable at fixed $N$ is therefore a property of the topology of
the parameter space, not of the penalty, the dictionary, or the image.**

**[A] Theorem 6 is exactly the big-$M$ relaxation, which makes the contrast operational.** The
standard mixed-integer formulation writes $|c_j|\le Mz_j$, $\sum_j z_j\le N$, $z\in\{0,1\}^D$;
relaxing to $z\in[0,1]^D$ and eliminating $z$ gives feasibility iff $\sum_j|c_j|/M\le N$ and
$|c_j|\le M$ — that is, the projection of the relaxed feasible set onto $c$ is precisely
$P_{N,M}$. So Theorem 6 says the big-$M$ LP relaxation is the tightest convex relaxation available,
and Theorem 3 says its measure-space analogue collapses to a plain mass ball. §7 measures what the
surviving discrete version is worth.

**[A] And this is a second, independent role for separation.** The BLASSO recovery literature needs
a minimum-separation condition on the *true* atoms to prove exact support recovery (main document
§8). The requirement here is unrelated to recovery: separation is what stops a minimizing sequence
from merging two atoms into one and thereby escaping the amplitude cap. Two different theories
demanding the same hypothesis for different reasons is worth noticing, and neither implies the
other.

---

## 5. Consequence for the certificate

The BLASSO

$$\min_{\mu\in\mathcal{M}(\Theta)}\ \tfrac12\|\Phi\mu-y\|^2+\lambda|\mu|(\Theta)
\tag{P$\lambda$}$$

has a dual certificate $\eta_\lambda=\Phi^*p_\lambda$, and $\hat\mu$ is optimal for
(P$\lambda$) if and only if $\|\eta_\lambda\|_\infty\le1$ and
$\eta_\lambda(\theta_i)=\mathrm{sign}(a_i)$ at each of its atoms. That is a genuine certificate — of optimality **over $\mathcal{M}(\Theta)$**.

**[A] Be precise about what it does and does not give.** Since $\mathcal{F}_N^{\,\tau}\subseteq
\mathcal{M}(\Theta)$, a certificate for (P$\lambda$) proves *a fortiori* that $\hat\mu$ minimizes
$J+\lambda|\cdot|$ over the $N$-sparse family too. What it does not give is optimality for $J$
*alone* over that family, which is (P0). The two differ by the penalty term, and that difference is
exactly the relaxation gap. So the certificate is not vacuous — it answers a different question
sharply.

**[A] The same applies to any method that solves (P$\lambda$) globally, however strong its
guarantee.** Conic Particle Gradient Descent comes with a global-convergence theorem for
(P$\lambda$) under its assumptions (main document §7.1). Grant the theorem in full and the
conclusion is unchanged: what is reached is the global optimum of a problem whose feasible set is,
by Theorem 3, the convexification of the $N$-sparse family for *every* $N$. A stronger convex
solver moves along this axis, not across it.

**[A] And no choice of $\lambda$ repairs this.** One might hope to tune $\lambda$ until the
penalized problem and (P0) agree at the desired $N$. By Theorem 3 the feasible set is the same
convex ball whatever $N$ is intended, so $\lambda$ is the only handle, and it controls mass rather
than count: two encodings with the same mass and different atom counts are scored identically by
$|\cdot|$. Empirically the gap does not close — §10.4 measures 6–409% at matched $N$ with both
sides solved exactly, and §10.13 finds the certificate value uninformative about (P0) optimality.
The certificate is a diagnostic instrument, and the reason is structural rather than numerical.

---

## 6. The objection that has to be answered

> *"The BLASSO minimizer is itself a finite sum of Diracs — representer theorems say so, and under
> a separation condition it recovers the true support exactly. The feasible set may have forgotten
> $N$, but the solution has not. So who cares?"*

This is the right objection and it is half correct. Two answers, and then a third route that
the objection itself suggests.

**[A] First: sparsity of the minimizer is not optimality at fixed $N$.** For a given $\lambda$ the
minimizer $\hat\mu_\lambda$ is generically a finite sum of Diracs — grant it. It minimizes
$J+\lambda|\cdot|$. The encoding problem asks for the minimizer of $J$ *alone* subject to at most
$N$ atoms. These coincide only if the penalty happens to be inactive at the constrained optimum,
which it is not; the difference is precisely the relaxation gap, and §10.4 measures it at
**6–409%** with $\ell_0$ solved by exhaustive enumeration and $\ell_1$ by the exact LARS path, so
neither number can be blamed on a solver. Getting an $N$-atom answer and getting the best $N$-atom
answer are different achievements, and only the first is on offer.

**[A] Second: exact-recovery guarantees do not apply to encoding.** They are statements about
in-model targets — the signal genuinely *is* a sparse measure, there is a true support, and
"recovery" has a referent — under a minimum-separation condition. An image is not exactly a
finite combination of Gaussians. There is no true support to recover, only approximation, and the
separation of the atoms in a good approximation is whatever the image dictates rather than
something one may assume. The main document measures both regimes separately (§9.1 in-model, §9.2
out-of-model) for exactly this reason.

**[A] A third route, and why it is not open either: impose separation directly.** Restricting to
measures whose atoms are $\delta$-separated *would* block the collision of Theorem 1 and rescue the
cap. But that set is not convex: $\delta_0$ and $\delta_{\delta/2}$ are each trivially separated,
while their average $\tfrac12\delta_0+\tfrac12\delta_{\delta/2}$ has two atoms at distance
$\delta/2$. Imposing separation is exactly what discretising onto a grid does — and it does it by
leaving the convex world, not by finding a cleverer convex set inside it.

---

## 7. What the measurements say

Six results from `optimal-encoding.md` and this repository, each an independent check on some part of the above.

**[V] The gap is real with every solver exact** (§10.4, `experiments/e4_exact_l0.py`). Shrink the
problem until $\ell_0$ can be solved by exhaustive enumeration of all $\binom{D}{N}$ supports and
$\ell_1$ by the exact LARS path, so neither side can be blamed on convergence. At matched $N\le4$,
$\ell_1$ is **6–409%** worse. Theorem 3 predicts a gap that no better convex penalty removes; this
measures it with the solver excuse eliminated.

**[V] Mass-constrained bounds are weak, as predicted** (§10.2, `experiments/e3_absolute_bound.py`).
A certified lower bound on the (P0) optimum built from a mass constraint — the tightest object any
measure-space convexification can see, by Theorem 3 — goes **vacuous by $N=4$–8**. At $N=1$, where
greedy is provably exactly optimal, the bound certifies only 24–34%.

**[V] The per-atom cap is informative on a grid and still numerically useless here** (§10.8,
`experiments/e8_branch_and_bound.py`). Theorem 6 says the big-$M$ constraint is not annihilated on
a finite dictionary, so branch-and-bound has something to exploit. It was measured: at the tightest
admissible box — $M$ just above the incumbent's own largest amplitude, below which the incumbent
becomes infeasible — the bound is loose by a factor of **6–9**, and 200,000 nodes returned a 100%
gap. The diagnosis is specific to this dictionary: the incumbent's largest amplitude is 24.0
against $\|y\|=21.2$, so **a single Gaussian carries the energy of the whole image**, forcing $M$ to
signal scale and $NM$ far above the true $\ell_1$ mass. **[A]** Note the two failures are
different: in the continuum the cap is destroyed by collision; on the grid it survives and is
merely far too generous.

**[V] The perspective relaxation fails the same way** (§10.9, `experiments/e9_perspective.py`). The
standard strengthening from the sparse-regression literature has root gap **64–86%** wherever the
ridge is small enough to leave the problem intact, and closes only at a ridge that makes the
encoding 2.4–4.3× worse. Two different norm-based relaxations failing for one shared reason —
amplitudes at signal scale — is evidence the obstruction is the splatting dictionary rather than
the choice of relaxation.

**[V] On a grid, the relaxation's tightness is governed by coherence — and coherence is what makes
the dictionary useful** (`experiments/e4_exact_l0.py`, raw output `results/e4_coherence.txt`).
Theorem 6 restores $N$-dependence on a finite dictionary but says nothing about how *tight* the
resulting relaxation is. Pruning the dictionary by greedy decorrelation and re-solving both problems
exactly at $N=3$ measures it:

| coherence | $D$ | exact $\ell_0$ error | $\ell_1$ excess over $\ell_0$ | shared support |
|---|---|---|---|---|
| 0.985 | 768 | 11.86% | **+145%** | 1/3 |
| 0.900 | 470 | 15.23% | +91% | 1/3 |
| 0.750 | 103 | 22.20% | +19% | 2/3 |
| 0.448 | 59 | **46.61%** | +5% | 2/3 |

(cartoon target; ascent and face behave the same way, reaching a 0% gap at coherence 0.599 with the
error at 41% against 23% for the full dictionary.) **[A] The convex relaxation can be made tight,
and the price is the dictionary's approximation power.** Coherence is not a defect to be engineered
away: a dictionary of near-orthogonal Gaussians cannot represent images at few atoms. So the
discrete route restores $N$-dependence in principle and delivers a tight relaxation only in the
regime where there is nothing worth approximating.

**[V] The certificate value carries no measurable information about (P0) optimality** (§10.13,
`experiments/e14_certifiable.py`). **[A]** Note what §5 does and does not predict: it says the
certificate cannot *prove* (P0) optimality, which leaves open that its value might still
*correlate* with it and serve as a heuristic. That is a separate, empirical question, and it was
asked separately. Across 178 distinct
local optima on 12 images, the $\lambda=0$ certificate value separates globally-optimal from
merely-locally-optimal solutions at pooled AUC **0.560**, against **0.363** for a null control that
carries no information by construction — the deviation from chance is the same size for both. **[A]** One qualification: that experiment maximizes
$|\eta(\theta)|$ over a 248-atom dictionary rather than over continuous $\theta$, so it measures a
lower bound on the true certificate value. The prediction it confirms is qualitative.

---

## 8. What is *not* claimed

**Not that (P0) is unsolvable.** The theorem says convexification over measures cannot encode $N$.
It says nothing about direct methods. On a finite dictionary at $N=3$, 1-swap local search from 100
random starts reached the exhaustively verified optimum on **40 of 40** images (§10.13).

**Not that the mass-ball relaxation is useless.** It is a valid lower bound on (P0) — just an
$N$-free one, and §10.2 measures how weak.

**Not that no convex formulation exists in any other space.** The result is about
$\mathcal{M}(\Theta)$ with the $N$-sparse family as the object being relaxed. **[U]** Moment–SOS
hierarchies over measures do encode support size, but through the *rank* of a moment matrix, which
is not a convex constraint; the convex relaxations in that family (fixed moment order) are subject
to the same collapse unless separation is imposed. This document does not test that route and no
source for it has been read here.

**Not that the discrete route is practically strong.** Theorem 6 restores $N$-dependence in
principle; §10.8 measures what it is worth on this dictionary and the answer is 6–9× loose.

**Hypotheses that matter.** Theorem 1 needs $N\ge2$ and an accumulation point; Theorem 3 needs
$\Theta$ compact and perfect; Theorem 6 needs $N$ integer and $N<D$; Corollary 5 needs $\varphi$
continuous. §9 says what happens when each fails.

---

## 9. How this could be wrong

The proofs are self-contained, so the exposed surface is the hypotheses and the measurements, not
the arguments. Precisely:

**The theorems fail if $\Theta$ is not perfect.** Theorem 1 needs an accumulation point to run the
collision, and Theorem 3 needs one at every point. A parameter space of isolated points is the
finite-dictionary case (§4), where the conclusion genuinely reverses. Any claim in this document
about the continuum is therefore a claim about a parameter space in which atoms may be placed
arbitrarily close together — which is what "continuous placement" means, but it is a hypothesis and
not a tautology.

**They fail if $\varphi$ is discontinuous on $\Theta$.** Weak-\* continuity of $J$ (§2), which
Corollary 5 rests on, needs $\theta\mapsto\varphi_\theta$ continuous. Scales bounded away from zero
give this. At $\sigma\to0$ the atom degenerates and continuity is lost — a different failure mode,
not covered here, and one that the CPGD guarantees in the main document (§7.1) also stumble over
for their own reasons.

**A concrete refutation would be:** a convex, weak-\* lower semicontinuous $R$ and a level $c$ with
$R\le c$ on $\mathcal{F}_{N,M}$ and $R>c$ somewhere on $B_{NM}$ — for a compact perfect $\Theta$.
Corollary 4 says this cannot exist, so producing one would locate an error in Lemma 2 or Theorem 1.

**The empirical claims of §7 are the softer half.** Every one of them is measured on a single
dictionary family, at $N\le4$ for the exact-$\ell_0$ comparison and $N=3$ for the certificate test.
The theory says the gap cannot be removed by a better convex penalty; it does not predict the
*size* of the gap, and the sizes reported here are properties of Gaussian splatting atoms on small
images, not universal constants. The diagnosis that "a single atom carries the energy of the whole
image" is likewise specific to this dictionary and would not transfer to, say, a wavelet frame with
uniformly bounded atom amplitudes.

---

## 10. Provenance

| claim | status | where |
|---|---|---|
| Theorems 1, 3, 6, Lemma 2, Corollaries 4, 5 | **[A]** derived here; proofs above, not imported | this document |
| Extreme points of the TV ball are signed Diracs | **[A]** standard; Lemma 2 proves what is needed without citing it | this document |
| BLASSO formulation and dual certificate | **[V]** | `papers/1811.06416v1.pdf`, main doc §2, §6 |
| $\ell_1$ vs $\ell_0$ at matched $N$, both exact | **[V]** | `experiments/e4_exact_l0.py`, main doc §10.4 |
| Mass-constrained bound goes vacuous | **[V]** | `experiments/e3_absolute_bound.py`, §10.2 |
| Big-$M$ loose by 6–9× | **[V]** | `experiments/e8_branch_and_bound.py`, §10.8 |
| Perspective root gap 64–86% | **[V]** | `experiments/e9_perspective.py`, §10.9 |
| Coherence governs relaxation tightness, and costs approximation power | **[V]** | `experiments/e4_exact_l0.py`, `results/e4_coherence.txt` |
| Certificate value uninformative about (P0) | **[V]** | `experiments/e14_certifiable.py`, §10.13 |
| CPGD's global-convergence theorem for (P$\lambda$) | **[V]** statement; **[A]** its irrelevance here | `papers/1907.10300v2.pdf`, main doc §7.1 |
| Branch-and-bound reaching $p\sim10^7$ | **[U]** | search summaries only; never read |
| Moment–SOS hierarchies | **[U]** | not read, not tested |
