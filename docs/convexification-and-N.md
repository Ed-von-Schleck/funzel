# Why convex relaxation loses the atom count

An image is to be written as a sum of $N$ Gaussian blobs. The standard way of turning that into a
convex problem destroys the number $N$: after convexification, two blobs of amplitude $M$ and one
blob of amplitude $2M$ become the same object. This document proves it, identifies the single
setting where it does not happen, and reports what the loss costs in measurements.

It is self-contained. The experiments it cites are in this repository under `experiments/`; the
companion document `optimal-encoding.md` has their full write-ups. Section 11 lists which claims
are proved here, which are measured here, and which are neither.

---

## 1. The encoding problem

An image $y$ has $P$ pixels. A *splat* is a Gaussian blob

$$\varphi_\theta(x)=\exp\!\big(-\tfrac12 (x-c)^{\!\top} A\,(x-c)\big),
\qquad \theta=(c,A)$$

with centre $c\in\mathbb{R}^2$ and a positive definite $2\times2$ matrix $A$ setting the width,
elongation and orientation. Write $\Theta$ for the set of allowed $\theta$ and $\varphi_\theta$
for the blob rendered onto the pixel grid, a vector in $\mathbb{R}^P$. The sparse-approximation
literature calls these blobs *atoms*, which is where the title's "atom count" comes from; this
document says blob.

Given a budget of $N$ blobs, choose their parameters and amplitudes to minimise squared error:

$$\min_{\theta_1,\dots,\theta_N\in\Theta,\ a\in\mathbb{R}^N}
\Big\|\,y-\sum_{i=1}^{N}a_i\varphi_{\theta_i}\Big\|^2
\tag{P0}$$

$N$ is the compression budget. Each blob costs a fixed number of stored numbers, so $N$ fixes the
file size, and "the best image at this file size" is a question about (P0)'s optimum.

(P0) is not convex, for two separate reasons. The error is a non-convex function of the centres —
slide a blob across the image and the error goes down, up, and down again. And the constraint "at
most $N$ blobs" is not a convex constraint: average a one-blob encoding with a different one-blob
encoding and you get a two-blob encoding.

In practice one places $N$ blobs at random, runs Adam, and accepts the result. Different starting
points give different answers, and nothing indicates how far any of them is from the best.

---

## 2. The convex substitute

There is a standard way to make a sparse problem like this convex. Let the encoding be carried by
a **measure** on the parameter space:

$$\mu=\sum_{i=1}^{N}a_i\delta_{\theta_i}$$

where $\delta_\theta$ is a unit point mass sitting at $\theta$. The rendering operator

$$\Phi\mu=\int_\Theta \varphi_\theta\,d\mu(\theta)$$

is **linear** in $\mu$, and applied to the $\mu$ above it returns exactly
$\sum_i a_i\varphi_{\theta_i}$. The unknown is now a single object $\mu$ instead of $N$ parameter
vectors, and the data term $\tfrac12\|\Phi\mu-y\|^2$ is convex in it. The non-convexity in the
centres is gone: a blob is no longer moved, it is *reweighted*, and moving a blob from $\theta$ to
$\theta'$ is a straight line in $\mu$ from $\delta_\theta$ to $\delta_{\theta'}$.

Sparsity is imposed with the **total variation norm** $|\mu|(\Theta)$, which for
$\mu=\sum_i a_i\delta_{\theta_i}$ equals $\sum_i|a_i|$. It is the continuum version of the $\ell_1$
norm, it is convex, and penalising it pushes mass onto few points. That gives

$$\min_{\mu}\ \tfrac12\|\Phi\mu-y\|^2+\lambda\,|\mu|(\Theta)
\tag{P$\lambda$}$$

known as the BLASSO. It is convex, so it has no local minima to be trapped in, and it comes with a
**dual certificate**. Given a candidate $\hat\mu$ with residual $p=y-\Phi\hat\mu$, form the
function $\eta(\theta)=\langle \varphi_\theta,\,p\rangle/\lambda$ on the parameter space: the
correlation of each possible blob with what is left over. If $|\eta|\le1$ everywhere and
$\eta(\theta_i)=\mathrm{sign}(a_i)$ at each blob actually used, then $\hat\mu$ is the global
minimiser. (P0) admits no such test.

The hope is then: solve (P$\lambda$), obtain a certified global optimum, and tune $\lambda$ until
the answer happens to use about $N$ blobs.

Notice what was traded. (P0) constrains the **number** of blobs. (P$\lambda$) penalises the
**total mass** and lets the number fall out of $\lambda$. The rest of this document is about what
that substitution costs.

---

## 3. What convexification does to a non-convex set

Convex methods cannot optimise over a non-convex set. Minimising $f$ over a set $S$ is *relaxed* by
replacing $S$ with a convex set containing it; the minimum over the larger set is a lower bound on
the minimum over $S$. The tightest possible choice is the **convex hull** of $S$, written
$\operatorname{conv}S$ — every point reachable as a weighted average of points of $S$. If one also
wants the minimum to be attained, one takes the closure, $\overline{\operatorname{conv}}\,S$.

So the best any convex approach can do with "at most $N$ blobs" is determined by the convex hull of
the set of $N$-blob encodings. **If that hull does not depend on $N$, then no convex method depends
on $N$** — not this penalty or that one, not a cleverer algorithm.

The hull does not depend on $N$. The reason is one picture. Take two blobs, each of amplitude $M$,
and slide them towards each other. Their masses add. In the limit they are a single blob of
amplitude $2M$ — an encoding that uses *one* blob, and that violates any per-blob amplitude limit
that the two original blobs respected. Nothing prevents this: the parameter space is continuous, so
two blobs can be brought arbitrarily close together.

Repeat with $N$ blobs of amplitude $M$ and the limit is one blob of amplitude $NM$. The consequence
is stated as Theorem 3 below: after convexification, the family of $N$-blob encodings with per-blob
amplitude at most $M$ is *exactly* the set of all measures of total mass at most $NM$, whatever
their number of blobs. Only the product $NM$ survives. Halve the blob budget, double the allowed
amplitude, and the convex problem does not change at all.

---

## 4. Notation

$\Theta$ is **compact**: centres in the closed image square, log-widths in a closed interval
bounded away from $\pm\infty$, orientation in a closed interval. Bounding widths away from zero is
what makes $\Theta$ compact in the log parametrisation, and it also keeps $\varphi$ continuous —
under pointwise sampling $\varphi_\theta(x)\to\mathbf 1\{x=c\}$ as the width goes to zero, a limit
that jumps depending on whether the centre lands exactly on a pixel, so $\theta\mapsto
\varphi_\theta$ has no continuous extension to zero width.

$\mathcal{M}(\Theta)$ is the space of finite signed measures on $\Theta$. It carries the **weak-\***
topology: $\mu_k\to\mu$ means $\int f\,d\mu_k\to\int f\,d\mu$ for every continuous $f$. This is the
notion of convergence in which a point mass can slide — $\delta_{\theta_k}\to\delta_\theta$ when
$\theta_k\to\theta$ — and it is also the one in which bounded sets are compact, so that minimising
sequences have limits.

Write

$$J(\mu)=\tfrac12\|\Phi\mu-y\|^2,
\qquad B_\rho=\{\mu: |\mu|(\Theta)\le\rho\}$$

$B_\rho$ is the set of encodings of total mass at most $\rho$, of any number of blobs.

**$J$ is weak-\* continuous.** Each pixel value $\mu\mapsto\int\varphi_\theta(x)\,d\mu(\theta)$ is
weak-\* continuous because $\varphi_\cdot(x)$ is a continuous function on $\Theta$; there are
finitely many pixels, so coordinatewise convergence is convergence in $\mathbb{R}^P$. (With a
continuum of pixels this would need an extra argument. Nothing here depends on that case.)

The two families of interest, for $N\ge1$ and $M>0$:

$$\mathcal{F}_{N,M}=\Big\{\sum_{i=1}^{N}a_i\delta_{\theta_i}:\ \theta_i\in\Theta,\ |a_i|\le M\Big\},
\qquad
\mathcal{F}_N^{\,\tau}=\Big\{\sum_{i=1}^{N}a_i\delta_{\theta_i}:\ \theta_i\in\Theta,\
\textstyle\sum_i|a_i|\le\tau\Big\}$$

Both are sums of at most $N$ point masses — some $a_i$ may be zero — with no smeared-out part, so
they are the $N$-blob encodings in the sense (P0) means. $\mathcal{F}_{N,M}$ limits **each** blob's
amplitude to $M$; $\mathcal{F}_N^{\,\tau}$ limits only the **total**.

---

## 5. The theorems

> **Theorem 1 (collision).** Let $\theta^\*$ be a point of $\Theta$ with other points of $\Theta$
> arbitrarily close to it, and let $N\ge2$. Then $NM\,\delta_{\theta^\*}$ is a weak-\* limit of
> elements of $\mathcal{F}_{N,M}$.

*Proof.* Choose for each $k$ a set of $N$ distinct points $\theta_1^k,\dots,\theta_N^k\in\Theta$
with $\theta_i^k\to\theta^\*$. Put $\mu_k=\sum_{i=1}^{N}M\delta_{\theta_i^k}$, which lies in
$\mathcal{F}_{N,M}$. For continuous $f$, $\int f\,d\mu_k=M\sum_i f(\theta_i^k)\to NMf(\theta^\*)$.
$\square$

Each $\mu_k$ obeys the per-blob limit $M$. The limit carries $NM$ at a single point. **The per-blob
limit does not survive.**

> **Lemma 2 (point masses generate the ball).** For any $\rho>0$,
> $\overline{\operatorname{conv}}\{\pm\rho\,\delta_\theta:\theta\in\Theta\}=B_\rho$.

*Proof.* ($\subseteq$) $B_\rho$ is convex, weak-\* closed, and contains every
$\pm\rho\delta_\theta$.

($\supseteq$) Let $|\mu|(\Theta)\le\rho$. Fix $m$, partition $\Theta$ into Borel pieces
$B_1,\dots,B_{k_m}$ of diameter below $1/m$, pick $\theta_j\in B_j$, set $a_j=\mu(B_j)$ and
$\mu_m=\sum_j a_j\delta_{\theta_j}$. For continuous $f$, uniform continuity gives

$$\Big|\int f\,d\mu_m-\int f\,d\mu\Big|
=\Big|\sum_j\int_{B_j}\big(f(\theta_j)-f(\theta)\big)d\mu(\theta)\Big|
\le \omega_f(1/m)\,\rho\ \longrightarrow\ 0$$

so $\mu_m\to\mu$ weak-\*, where $\omega_f$ is the modulus of continuity of $f$. Also
$\sum_j|a_j|\le|\mu|(\Theta)\le\rho$, so setting $\lambda_j=|a_j|/\rho$ and
$\lambda_0=1-\sum_j\lambda_j\ge0$,

$$\mu_m=\sum_j\lambda_j\big(\rho\,\mathrm{sign}(a_j)\delta_{\theta_j}\big)+\lambda_0\cdot0,
\qquad 0=\tfrac12(\rho\delta_{\theta_1})+\tfrac12(-\rho\delta_{\theta_1})$$

writes $\mu_m$ as a finite weighted average of points $\pm\rho\delta_\theta$. So
$\mu\in\overline{\operatorname{conv}}\{\pm\rho\delta_\theta\}$. $\square$

This is Krein–Milman applied to $B_\rho$, whose extreme points are the signed point masses. The
proof above is included so that the document does not depend on a theorem it has not stated.

> **Theorem 3 (the atom count is lost).** Let $\Theta$ be compact, with every point of $\Theta$
> having other points of $\Theta$ arbitrarily close to it. Then for every $N\ge1$ and $M>0$
> $$\overline{\operatorname{conv}}\,\mathcal{F}_{N,M}=B_{NM},
> \qquad
> \overline{\operatorname{conv}}\,\mathcal{F}_N^{\,\tau}=B_\tau .$$

*Proof.* Every element of $\mathcal{F}_{N,M}$ has total mass at most $NM$, and $B_{NM}$ is convex
and weak-\* closed, which gives $\subseteq$. For $\supseteq$, fix $\theta\in\Theta$. If $N=1$ then
$\pm M\delta_\theta$ is already in $\mathcal{F}_{1,M}$. If $N\ge2$, Theorem 1 puts
$NM\delta_\theta$ in the closure of $\mathcal{F}_{N,M}$, and $-NM\delta_\theta$ likewise by taking
every $a_i=-M$. Either way $\pm NM\delta_\theta$ lies in the closure for every $\theta$, so Lemma 2
with $\rho=NM$ gives $B_{NM}\subseteq\overline{\operatorname{conv}}\,\mathcal{F}_{N,M}$. The second
identity is the case $M=\tau$, $N=1$, together with
$\mathcal{F}_1^{\,\tau}\subseteq\mathcal{F}_N^{\,\tau}\subseteq B_\tau$. $\square$

The hypothesis holds for splatting: $\Theta$ is a product of intervals, so every point has
neighbours arbitrarily close. It fails for a finite dictionary, which is Section 7.

> **Corollary 4 (no convex penalty separates).** Let $R$ be convex and weak-\* lower
> semicontinuous. If $R\le c$ everywhere on $\mathcal{F}_{N,M}$, then $R\le c$ everywhere on
> $B_{NM}$.

*Proof.* $\{R\le c\}$ is convex and weak-\* closed, so it contains
$\overline{\operatorname{conv}}\,\mathcal{F}_{N,M}=B_{NM}$. $\square$

No convex regulariser — not $\ell_1$, not a reweighted or spatially varying or group-structured
variant — has a level set that accepts the $N$-blob encodings and rejects a dense measure of the
same total mass.

Lower semicontinuity is a real hypothesis: a convex function whose level sets are not closed is not
covered by Corollary 4. It is not a way around the result, because Corollary 5 reaches the same
conclusion about *optimal values* without assuming anything is closed.

> **Corollary 5 (the values collapse too).**
> $\displaystyle\inf_{\mu\in\operatorname{conv}\mathcal{F}_{N,M}}J(\mu)
> =\inf_{\mu\in B_{NM}}J(\mu)$.

*Proof.* $J$ is weak-\* continuous, so its infimum over a set equals its infimum over that set's
closure. Apply Theorem 3. $\square$

The plain convex hull, before any closure, is genuinely smaller than the ball: a finite weighted
average $\sum_j\lambda_j\mu_j$ of elements of $\mathcal{F}_{N,M}$ has mass at most $M$ at any
single point, so the per-blob limit does hold there.
It makes no difference. The *infimum* over that smaller set is already the infimum over the whole
ball. Colliding blobs are not an artefact of insisting on closed sets — minimising sequences run
into them.

---

## 6. Four ways out, and why each closes

**Limit each blob's amplitude.** This is the natural repair and it is what every mixed-integer
sparse solver does, under the name big-$M$. Theorem 1 kills it: blobs collide and the limit
exceeds the cap. Section 7 shows the cap does work when the parameter space is finite, and Section
9 measures what it is worth there.

**Change the topology.** Theorem 3 closes in the weak-\* sense, and the choice matters: in the
total-variation *norm*, $\mu_k\to\mu$ forces $\mu_k(\{\theta\})\to\mu(\{\theta\})$ at each point,
so the cap would survive and the norm-closed hull would be strictly smaller than the ball. One
could hope to build a relaxation there. Corollary 5 blocks it, because the collapse it states is
about the **un-closed** hull, where no topology appears. Whichever topology is preferred, the best
value a convex relaxation of the capped $N$-blob problem can return is the value of the plain
mass-ball problem. (Weak-\* is also the natural choice: bounded sets are weak-\* compact and are not
TV-compact, so minimising sequences have weak-\* limits and generally no TV limit.)

**Forbid blobs from colliding.** Restricting to encodings whose blobs are at least $\delta$ apart
does block Theorem 1. That set is not convex: $\delta_0$ and $\delta_{\delta/2}$ are each a single
blob and trivially "separated", while their average $\tfrac12\delta_0+\tfrac12\delta_{\delta/2}$ has
two blobs at distance $\delta/2$. Imposing separation is what putting the blobs on a fixed grid
does, and it does it by leaving the convex setting rather than by finding a better convex set
inside it.

**Point out that the BLASSO's own solutions are sparse.** They are: for a fixed $\lambda$ the
minimiser of (P$\lambda$) is generically a finite sum of point masses, and under a separation
condition it recovers a true sparse signal exactly. Two things are still wrong with using that as
an answer.

First, a sparse minimiser is not an optimal $N$-blob encoding. It minimises
$J+\lambda|\cdot|$. (P0) asks for the minimiser of $J$ alone under a blob budget. They agree only if
the penalty is inactive at the constrained optimum, which it is not, and the difference is the
relaxation gap — measured in Section 9 at 6–409% with both problems solved exactly.

Second, exact-recovery results are about signals that genuinely *are* sparse combinations of blobs,
where there is a true support to recover and a separation between its atoms to assume. An image is
not a finite combination of Gaussians. There is no true support, only approximation, and the
spacing of blobs in a good approximation is whatever the image dictates.

---

## 7. Where the argument fails: finite dictionaries

Fix a finite list of $D$ candidate blobs and let the encoding be a coefficient vector
$c\in\mathbb{R}^D$, with $\|c\|_0$ the number of non-zeros. Now blobs cannot collide: the
candidates sit at fixed, distinct positions.

> **Theorem 6.** Let $1\le N<D$ with $N$ an integer, and $M>0$. Then
> $$\operatorname{conv}\{c:\|c\|_0\le N,\ \|c\|_\infty\le M\}
> =\{c:\|c\|_1\le NM\}\cap\{c:\|c\|_\infty\le M\}=:P_{N,M},$$
> and for $N\ge2$ this is strictly smaller than $\{c:\|c\|_1\le NM\}$.

*Proof.* ($\subseteq$) An $N$-sparse $c$ with $\|c\|_\infty\le M$ has $\|c\|_1\le NM$, and both
constraints are convex.

($\supseteq$) $P_{N,M}$ is a bounded polytope, so it is the convex hull of its vertices, and it is
enough to show every vertex is $N$-sparse with entries $\pm M$. Let $c$ be a vertex and
$k=\#\{j:|c_j|=M\}$. Suppose $\|c\|_1<NM$. Then the only active constraints are of the form
$c_j=\pm M$, and a vertex in $\mathbb{R}^D$ needs $D$ independent active constraints, forcing
$|c_j|=M$ for every $j$ and so $\|c\|_1=DM>NM$ — impossible since $N<D$. So $\|c\|_1=NM$, and the
coordinates with $0<|c_j|<M$ carry $(N-k)M$ between them. If two of them were non-zero, shifting
$t$ from one to the other leaves both $\|c\|_1$ and $\|c\|_\infty$ unchanged, and $c$ would be the
midpoint of two feasible points, so not a vertex. So at most one such coordinate exists; if it does,
it equals $(N-k)M$, and $0<(N-k)M<M$ forces $0<N-k<1$, impossible for integers. Hence no coordinate
lies strictly between, $\|c\|_1=kM=NM$, and $k=N$.

(Strictness) For $N\ge2$, the vector with $NM$ in one coordinate has $\|c\|_1=NM$ but
$\|c\|_\infty=NM>M$. $\square$

$P_{N,M}$ depends on $N$ and $M$ separately, not only on their product. That is the whole
difference from Theorem 3, and its cause is that coordinates cannot merge.

**$P_{N,M}$ is exactly the big-$M$ relaxation.** The standard mixed-integer formulation writes
$|c_j|\le Mz_j$ with $\sum_j z_j\le N$ and $z_j\in\{0,1\}$. Relaxing to $z_j\in[0,1]$ and
eliminating $z$ leaves $\sum_j|c_j|/M\le N$ and $|c_j|\le M$ — that is, $P_{N,M}$. So the
relaxation that branch-and-bound solvers actually use is the tightest convex relaxation there is,
and its continuum counterpart collapses to a plain mass ball.

**A compact space in which every point is isolated is finite.** So a compact parameter space with
no isolated points falls under Theorem 3 and one with only isolated points is finite and falls under
Theorem 6. Spaces that mix the two are not treated here and do not arise in this work. Whether the
blob count can be represented convexly is a property of the shape of the parameter space, not of the
penalty, the dictionary, or the image.

The same separation hypothesis appears in the recovery theory for the BLASSO, where the true blobs
must be a minimum distance apart for exact recovery to be provable. The requirement here is
unrelated to recovery: separation is what stops a minimising sequence from merging two blobs into
one. Two theories needing the same hypothesis for different reasons; neither implies the other.

---

## 8. What this means for the certificate

The BLASSO's dual certificate proves that a candidate $\hat\mu$ minimises (P$\lambda$) over all of
$\mathcal{M}(\Theta)$. That is a genuine statement, but note which problem it is about.

Since the $N$-blob encodings are a subset of $\mathcal{M}(\Theta)$, the certificate does prove that
$\hat\mu$ minimises $J+\lambda|\cdot|$ over them as well. What it does not give is optimality for
$J$ alone under a blob budget, which is (P0). The two objectives differ by the penalty term, and
that difference is the relaxation gap.

Tuning $\lambda$ does not fix it. By Theorem 3 the feasible set is the same ball regardless of what
$N$ was intended, so $\lambda$ is the only remaining handle, and $\lambda$ controls mass, not count:
two encodings with the same mass and different blob counts are scored identically by
$|\cdot|(\Theta)$.

Nor does a better solver. Conic Particle Gradient Descent, for instance, comes with a global
convergence theorem for (P$\lambda$) under its assumptions. Grant the theorem in full and the
conclusion is unchanged: what is reached is the global optimum of a problem whose feasible set is,
by Theorem 3, the convexification of the $N$-blob family for *every* $N$ at once.

---

## 9. What the measurements show

Six results from this repository. Each is described here in enough detail to be read without the
other document.

**The gap is real with both problems solved exactly.** `experiments/e4_exact_l0.py`. Shrink the
problem until neither side can be blamed on a solver: a $32\times32$ image, a dictionary of $D=768$
blobs, and $N\le4$. Solve $\ell_0$ by enumerating every one of the $\binom{768}{3}=75{,}202{,}816$
supports, and $\ell_1$ by following its exact solution path as $\lambda$ decreases — the path is
piecewise
linear, so it can be computed without approximation — then re-fitting amplitudes on the blobs it
selects, which removes the shrinkage bias and can only help it. At matched blob count $\ell_1$ is
worse by between **6% and 409%** across four targets and budgets from 2 to 5 blobs. The two agree
only at $N=1$, where the problems coincide. The worst case, 409%, is on a target built as an exact
sum of three dictionary blobs, at $N=2$: $\ell_0$ reaches error 8.70 and $\ell_1$ 44.25. On a
cartoon target at $N=2$ the figures are 18.00 and 30.90, and the two answers share no blob at all.
Theorem 3 says no better convex penalty removes this; the experiment says how
large it is when nothing is approximate.

**A mass-constrained bound goes vacuous quickly.** `experiments/e3_absolute_bound.py`. Theorem 3
says a total-mass constraint is the most any measure-space convexification can see, so a lower
bound on (P0) built from one is the best of its kind. Measured on $64\times64$ images against
budgets 1 to 16, it stops saying anything by $N=4$–8, where it falls to zero. At $N=1$, where the
single best blob can be found exactly by direct search so the true
optimum is known, the bound accounts for only 24–34% of it.

**The per-blob cap survives discretisation and is still useless here.**
`experiments/e8_branch_and_bound.py`. On a finite dictionary Theorem 6 applies, so branch-and-bound
has something to work with. Implemented on a $48\times48$ image with $D=248$: the node bound is
informative only when a certain ratio falls below 1, and at the tightest cap the search may legally
use — $M$ set to the largest amplitude in the best solution found so far, below which that solution
would itself be excluded — the ratio is **5.7 at $N=4$ and 8.9 at $N=6$**. Two hundred thousand
nodes returned a 100% gap.
The reason is specific to Gaussian blobs: the incumbent's largest amplitude is 24.0 while
$\|y\|=21.2$, so **a single blob carries the energy of the whole image**, and any cap large enough
to admit it is far too large to constrain $N$ blobs. This is a different failure from the
continuum one. There the cap is destroyed by collision; here it survives and is merely far too
generous.

**The perspective relaxation fails the same way.** `experiments/e9_perspective.py`. The standard
strengthening from the sparse-regression literature, applied at $D=248$, $N=3$ on $48\times48$
images. It only applies to a modified problem carrying an extra penalty $\lambda_2\|c\|^2$ on the
amplitudes. Where that penalty is small enough to leave the problem essentially unchanged
($\lambda_2\le10^{-3}$, reconstruction identical to four decimals) the gap between the bound and
the true optimum, before any branching has happened, is **64–86%**. It closes only at
$\lambda_2=1$, where the
reconstruction error has risen by a factor of 2.4–4.3. There is no setting in which the bound is
tight and the problem is still the one wanted. Two relaxations failing for one shared reason —
amplitudes at signal scale — points at the dictionary rather than at the choice of relaxation.

**On a grid, the relaxation is tight only where the dictionary is useless.**
`experiments/e4_exact_l0.py`, output `results/e4_coherence.txt`. Theorem 6 restores the dependence
on $N$ but says nothing about how *tight* the resulting relaxation is. Pruning the dictionary by
greedy decorrelation and re-solving both problems exactly at $N=3$ on a $32\times32$ image:

| coherence | $D$ | exact $\ell_0$ error | $\ell_1$ excess | shared support |
|---|---|---|---|---|
| 0.985 | 768 | 11.86% | **+145%** | 1 of 3 |
| 0.900 | 470 | 15.23% | +91% | 1 of 3 |
| 0.750 | 103 | 22.20% | +19% | 2 of 3 |
| 0.448 | 59 | **46.61%** | +5% | 2 of 3 |

Coherence is the largest inner product between two normalised dictionary blobs; $\ell_1$ theory
wants it well below 1 and here it starts at 0.985. Decorrelating the dictionary does close the
$\ell_1$–$\ell_0$ gap, and it destroys the dictionary's ability to approximate: the error at
coherence 0.448 is four times the error at 0.985. Near-orthogonal Gaussians cannot represent an
image in few blobs. The convex relaxation becomes tight in the regime where there is nothing worth
approximating.

**The certificate's value does not indicate (P0) optimality either.**
`experiments/e14_certifiable.py`. Section 8 says the certificate cannot *prove* (P0) optimality.
Its numerical value might still correlate with being optimal and serve as a heuristic, which is a
separate question. Across 178 distinct local optima
on 12 images at $N=3$, the certificate value at $\lambda=0$ separates the best solution from the
rest with AUC **0.560** — the probability that it ranks the best solution ahead of a randomly
chosen worse one, where 0.5 is chance — against **0.363** for a control drawn from a random number
generator, which carries no information by construction. Both sit about 0.14 from chance, so the
signal is the size of the noise. One qualification: that experiment maximises the certificate over a 248-blob
dictionary rather than over all of $\Theta$, so it measures a lower bound on the true certificate
value.

---

## 10. Limits

The proofs are self-contained, so what is exposed is the hypotheses and the measurements.

**The theorems need blobs to be able to collide.** Theorem 1 needs points of $\Theta$ arbitrarily
close to one another, and Theorem 3 needs that everywhere. A parameter space of isolated points is
the finite-dictionary case of Section 7, where the conclusion reverses. Every claim here about the
continuum is a claim about a parameter space in which blobs can be placed arbitrarily close
together. That is what continuous placement means, but it is an assumption and is stated as one.

**They need $\varphi$ continuous on $\Theta$.** Corollary 5 rests on the weak-\* continuity of $J$,
which needs $\theta\mapsto\varphi_\theta$ continuous. Widths bounded away from zero give this. At
zero width the blob degenerates and continuity is lost — a different failure, not covered here.

**A concrete refutation would be** a convex, weak-\* lower semicontinuous $R$ and a level $c$ with
$R\le c$ on $\mathcal{F}_{N,M}$ and $R>c$ somewhere on $B_{NM}$, for a compact $\Theta$ with no
isolated points. Corollary 4 says this cannot exist, so exhibiting one would locate an error in
Lemma 2 or Theorem 1.

**The measurements are less secure than the proofs.** All of them are on one dictionary family, at
$N\le4$ for
the exact comparison and $N=3$ for the certificate test, on images of 32 to 64 pixels a side. The
theory says the gap cannot be removed by a better convex penalty; it does not predict the gap's
*size*, and the sizes above are properties of Gaussian blobs on small images, not universal
constants. In particular "a single blob carries the energy of the whole image" is a fact about this
dictionary and would not hold for, say, a wavelet frame with uniformly bounded atom amplitudes.

**Not claimed: that (P0) is unsolvable.** The result is about convexification. Direct methods are
untouched, and on a finite dictionary at $N=3$, local search from 100 random starts reached the
exhaustively verified optimum on 40 of 40 images.

**Not claimed: that the mass-ball relaxation is worthless.** It is a valid lower bound on (P0).
It is simply blind to $N$, and the second measurement above shows how weak that makes it.

**Not claimed: that no convex formulation exists anywhere.** The result concerns
$\mathcal{M}(\Theta)$ with the $N$-blob family as the object being relaxed. Moment-based
hierarchies do encode support size, but through the rank of a moment matrix, which is not a convex
constraint. That route is neither tested here nor read up on.

---

## 11. Status of each claim

| claim | status |
|---|---|
| Theorems 1, 3, 6; Lemma 2; Corollaries 4, 5 | proved above; nothing imported |
| Extreme points of the total-variation ball are the signed point masses | standard; Lemma 2 proves what is used, so it is not relied on |
| BLASSO formulation and its dual certificate | from `papers/1811.06416v1.pdf` |
| Global convergence of Conic Particle Gradient Descent for (P$\lambda$) | statement from `papers/1907.10300v2.pdf`; its irrelevance here is the argument of Section 8 |
| $\ell_1$ versus $\ell_0$ at matched $N$, both exact | measured, `experiments/e4_exact_l0.py` |
| Mass-constrained bound goes vacuous by $N=4$–8 | measured, `experiments/e3_absolute_bound.py` |
| Big-$M$ node bound off by 5.7–8.9× | measured, `experiments/e8_branch_and_bound.py` |
| Perspective relaxation root gap 64–86% | measured, `experiments/e9_perspective.py` |
| Coherence governs tightness and costs approximation power | measured, `results/e4_coherence.txt` |
| Certificate value uninformative about (P0) optimality | measured, `experiments/e14_certifiable.py` |
| Branch-and-bound solvers reaching $10^7$ variables | from search summaries; the sources were never read |
| Moment-based hierarchies | not read, not tested |
