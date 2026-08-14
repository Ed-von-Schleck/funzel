# Why convex relaxation loses the atom count

An image is to be written as a sum of $N$ Gaussian blobs. The standard way of turning that into a
convex problem destroys the number $N$: after convexification, two blobs of amplitude $M$ and one
blob of amplitude $2M$ become the same object. This document proves that, identifies the three
places where it does not happen, bounds the damage at a fixed mass budget, and measures what the
loss is worth in practice.

The collapse is not a fact about convexity alone. It needs four things together: a convex feasible
set, a continuum of blob parameters, a rendering linear in the encoding, and an objective that sees
the encoding only through the rendered image. Remove any one and the count survives. A fixed
dictionary removes the continuum (Section 7). Requiring an encoding's blobs to stay apart removes
the collision the continuum allows (Section 6). Lifting to second moments removes the objective's
form, and convexification then costs nothing at all (Section 5.2). Each escape is paid for — with a
grid, with a hypothesis about which encodings count, or with any usable description of the feasible
set — and which of those is the real price is most of what follows.

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

$N$ is the compression budget: each blob costs a fixed number of stored numbers, so $N$ stands in
for the file size and "the best image at this size" becomes a question about (P0)'s optimum. That
is an idealisation. Amplitudes here span a wide range — Section 9 finds a single blob carrying the
energy of a whole image — and under any real quantiser their precision costs bits that a blob count
does not see. Everything below takes the count as the budget, as (P0) does; whether it is the right
budget is not examined here.

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

One handle on the count survives, and Carathéodory's theorem gives it. Take any non-zero $\mu$ and
write it as $s|\mu|$ with $s=\pm1$. Then $\Phi\mu=|\mu|(\Theta)\,\mathbb{E}[s(\theta)\varphi_\theta]$,
the expectation taken against $|\mu|/|\mu|(\Theta)$, so $\Phi\mu$ lies in $|\mu|(\Theta)$ times the
convex hull of $\{\pm\varphi_\theta:\theta\in\Theta\}$, a compact subset of $\mathbb{R}^P$.
Carathéodory writes any point of that hull as a combination of at most $P+1$ of its points, so some
$\nu$ with at most $P+1$ blobs has $\Phi\nu=\Phi\mu$ and mass no larger. It renders the same image
at no greater cost in either term, so it does at least as well in (P$\lambda$), and at least as well
under any mass budget. Some minimiser therefore uses at most $P+1$ blobs. That is the only bound on
the count convexification leaves standing, and it is set by the image size rather than by $\lambda$
— $P$ is the pixel count, so at any $N$ worth encoding at, the bound is true and idle.

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
the set of $N$-blob encodings. **If that hull does not depend on $N$, then no convex method that
minimises the error of the rendered image depends on $N$** — not this penalty or that one, not a
cleverer algorithm, and by Corollary 7 not a reformulation introducing new variables either. The
qualification is load-bearing rather than decorative: Theorem 9 exhibits a convex problem, with a
different objective, whose value equals the $N$-blob optimum for every $N$.

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
sequences have limits. $\Theta$ is compact metric, so $C(\Theta)$ is separable and the weak-\*
topology is metrisable on bounded sets; every closure below is taken inside a bounded set, so
sequences suffice to compute it.

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

A cap is needed for there to be anything to prove. (P0) lets $a$ range over $\mathbb{R}^N$, so its
$N$-blob family contains $\pm\rho\,\delta_\theta$ for every $\rho$, and Lemma 2 then puts every
$B_\rho$ in its closed convex hull: the hull is all of $\mathcal{M}(\Theta)$, vacuous for every $N$
at once. These two families are the simplest caps; Theorem 4 covers the rest.

---

## 5. The theorems

### 5.1 The collapse

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

*Proof.* ($\subseteq$) $B_\rho$ is convex and contains every $\pm\rho\delta_\theta$, and it is
weak-\* closed because $|\cdot|(\Theta)$ is the dual norm of the sup norm on $C(\Theta)$, hence
weak-\* lower semicontinuous.

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

The two identities are not equally substantial. The second is a restatement of a definition: the
total-variation norm is the atomic norm generated by the point masses, so its ball is by
construction the closed convex hull of them, which is Lemma 2, and $\mathcal{F}_1^{\,\tau}$ already
contains every $\pm\tau\delta_\theta$. Nothing is left to prove and nobody would cite a source for
it. The first identity is the content. Its constraint is on each amplitude separately rather than on
the sum, so it is not a gauge and the atomic-norm argument says nothing about it; reaching
$NM\delta_\theta$, which breaks the per-blob cap by a factor of $N$, needs the collision. Theorem 4
is the strong form.

The hypothesis holds for splatting: $\Theta$ is a product of intervals, so every point has
neighbours arbitrarily close. It fails for a finite dictionary, which is Section 7.

The $\ell^\infty$ cap in $\mathcal{F}_{N,M}$ is not what drives this. Any cap does the same.

> **Theorem 4 (the shape of the cap does not matter).** Let $A\subseteq\mathbb{R}^N$ be bounded
> and sign-symmetric — $a\in A$ implies $(\varepsilon_1a_1,\dots,\varepsilon_Na_N)\in A$ for every
> $\varepsilon\in\{\pm1\}^N$ — and put
> $$\mathcal{F}_A=\Big\{\sum_{i=1}^{N}a_i\delta_{\theta_i}:\theta_i\in\Theta,\ a\in A\Big\},
> \qquad \rho(A)=\sup_{a\in A}\|a\|_1 .$$
> Then, under the hypothesis of Theorem 3, $\overline{\operatorname{conv}}\,\mathcal{F}_A
> =B_{\rho(A)}$.

*Proof.* ($\subseteq$) Every element has total mass at most $\rho(A)$, and $B_{\rho(A)}$ is convex
and weak-\* closed. ($\supseteq$) Fix $\theta\in\Theta$ and $\varepsilon>0$, and pick $a\in A$ with
$\|a\|_1\ge\rho(A)-\varepsilon$. Sign-symmetry lets us take every $a_i\ge0$, so
$\sum_ia_i=\|a\|_1$. Choose distinct $\theta_1^k,\dots,\theta_N^k\in\Theta$ with
$\theta_i^k\to\theta$; then $\sum_ia_i\delta_{\theta_i^k}\in\mathcal{F}_A$ converges weak-\* to
$\|a\|_1\delta_\theta$, and flipping every sign gives $-\|a\|_1\delta_\theta$. Letting
$\varepsilon\to0$ puts $\pm\rho(A)\delta_\theta$ in the closure for every $\theta$, and Lemma 2 with
$\rho=\rho(A)$ finishes. $\square$

Theorem 3 is the case $A=\{\|a\|_\infty\le M\}$, with $\rho(A)=NM$; an $\ell^2$ cap $\|a\|_2\le1$
gives $\sqrt N$, an $\ell^1$ cap gives $\tau$. Each returns a mass ball, differing only in radius.
The contrast is with the finite setting, where the shape of the cap does change the answer. Take
$\operatorname{conv}\{c\in\mathbb{R}^D:\|c\|_0\le k,\ \|c\|_2\le1\}$. At $k=1$ the set is the union
of the coordinate segments $\{te_j:|t|\le1\}$ and its hull is the $\ell^1$ ball; at $k=D$ the set is
the $\ell^2$ ball already. For $D\ge2$ those differ, so the hull depends on $k$. That dependence
does not survive collision.

> **Corollary 5 (no convex penalty separates).** Let $R$ be convex and weak-\* lower
> semicontinuous, meaning each sublevel set $\{R\le c\}$ is weak-\* closed. If $R\le c$ everywhere
> on $\mathcal{F}_{N,M}$, then $R\le c$ everywhere on $B_{NM}$.

*Proof.* $\{R\le c\}$ is convex and weak-\* closed, so it contains
$\overline{\operatorname{conv}}\,\mathcal{F}_{N,M}=B_{NM}$. $\square$

No convex regulariser — not $\ell_1$, not a reweighted or spatially varying or group-structured
variant — has a level set that accepts the $N$-blob encodings and rejects a dense measure of the
same total mass.

Lower semicontinuity is a real hypothesis: a convex function whose level sets are not closed is not
covered by Corollary 5. It is not a way around the result, because Corollary 6 reaches the same
conclusion about *optimal values* without assuming anything is closed.

> **Corollary 6 (the values collapse too).**
> $\displaystyle\inf_{\mu\in\operatorname{conv}\mathcal{F}_{N,M}}J(\mu)
> =\inf_{\mu\in B_{NM}}J(\mu)$.

*Proof.* $J$ is weak-\* continuous, so its infimum over a set equals its infimum over that set's
closure. Apply Theorem 3. $\square$

The proof uses nothing about $J$ beyond weak-\* continuity, so every weak-\* continuous objective
has the same infimum over $\operatorname{conv}\mathcal{F}_{N,M}$ as over $B_{NM}$. In measure space
the collapse is a property of the feasible set; changing what is minimised does not affect it. The
objective matters only outside measure space, in Corollary 7, and Theorem 9 shows how far that
goes.

The plain convex hull, before any closure, is genuinely smaller than the ball: a finite weighted
average $\sum_j\lambda_j\mu_j$ of elements of $\mathcal{F}_{N,M}$ has mass at most $M$ at any single
point, so the per-blob limit does hold there. It makes no difference: the *infimum* over that
smaller set is already the infimum over the whole ball. Colliding blobs are not an artefact of
insisting on closed sets — minimising sequences run into them.

Corollary 5 restricts what a penalty on $\mu$ can do. The same collapse restricts what any convex
reformulation can do, including ones that introduce new variables.

> **Corollary 7 (no convex lift helps).** Let $V$ be a real vector space, $C\subseteq V$ convex,
> and $L:V\to\mathbb{R}^P$ linear, with $\Phi\mathcal{F}_{N,M}\subseteq L(C)$. Write
> $j(z)=\tfrac12\|z-y\|^2$. Then
> $$\inf_{v\in C}\,j(Lv)\ \le\ \inf_{\mu\in B_{NM}}J(\mu).$$

*Proof.* $L(C)$ is convex, being a linear image of a convex set, and it contains
$\Phi\mathcal{F}_{N,M}$, so it contains $\operatorname{conv}\Phi\mathcal{F}_{N,M}
=\Phi(\operatorname{conv}\mathcal{F}_{N,M})$. Hence $\inf_{v\in C}j(Lv)=\inf_{z\in L(C)}j(z)
\le\inf_{\mu\in\operatorname{conv}\mathcal{F}_{N,M}}J(\mu)$, which is $\inf_{B_{NM}}J$ by
Corollary 6. $\square$

Containing the $N$-blob encodings is what makes a relaxation a relaxation, and a linear image of a
convex set is convex. So the collapse does not come from having chosen $\mathcal{M}(\Theta)$: extra
variables cannot escape it, at any level of any hierarchy, as long as the feasible set is convex and
the rendered image is linear in them. No topology enters.

The hypothesis to attack is $\Phi\mathcal{F}_{N,M}\subseteq L(C)$. A bounding scheme that does not
contain the $N$-blob encodings is outside the statement as written — a Lagrangian dual, for
instance, is not obtained by enlarging the feasible set, and nothing here computes what one
returns.

Two exclusions, both real. A lift whose objective is not a function of the rendered image alone —
the perspective relaxation of Section 9, which carries $\lambda_2\|c\|^2$ on the amplitudes. And a
lift with a non-convex constraint, which is where the blob count lives in a moment hierarchy:
moment-matrix entries are linear in $\mu$, so convex constraints built from them leave a convex
lift, and cardinality enters as $\operatorname{rank}M_d(\mu)\le N$. Corollary 7 is why that
ingredient has
to be the non-convex one.

### 5.2 How large the collapse is, and where it stops

Theorem 3 is about the feasible set. It says the mass ball cannot see $N$; it does not say the
ball's optimal *value* is far from the $N$-blob optimum. At a fixed mass budget it provably is
not.

> **Theorem 8 (how much the mass ball can lose).** Let $b=\max_{\theta\in\Theta}\|\varphi_\theta\|$.
> For every $\rho>0$, every $\mu\in B_\rho$ and every $N\ge1$ there is $\nu\in
> \mathcal{F}_N^{\,\rho}$ with $\|\Phi\mu-\Phi\nu\|\le\rho b/\sqrt N$. Consequently
> $$\inf_{\mathcal{F}_N^{\,\rho}}J\ \le\ \tfrac12\Big(\sqrt{2\inf_{B_\rho}J}\ +\ \rho b/\sqrt N\Big)^{2}.$$

*Proof.* Put $t=|\mu|(\Theta)/\rho\le1$ and let $s=d\mu/d|\mu|\in\{\pm1\}$ be the sign of $\mu$,
that is, the $\pm1$ density of $\mu$ against its total-variation measure. Fix
any $\theta_0$ and let $\pi$ be the distribution on $G=\{\pm\varphi_\theta:\theta\in\Theta\}$ that
draws $s(\theta)\varphi_\theta$ with $\theta$ distributed as $|\mu|/|\mu|(\Theta)$ with probability
$t$, and $+\varphi_{\theta_0}$ or $-\varphi_{\theta_0}$ with probability $(1-t)/2$ each. Its mean is
$t\,\Phi\mu/|\mu|(\Theta)=\Phi\mu/\rho$. Draw $h_1,\dots,h_N$ independently from $\pi$ and set
$\hat g=\tfrac1N\sum_ih_i$. Then $\mathbb{E}\hat g=\Phi\mu/\rho$ and

$$\mathbb{E}\big\|\hat g-\Phi\mu/\rho\big\|^2
=\tfrac1N\Big(\mathbb{E}\|h\|^2-\|\Phi\mu/\rho\|^2\Big)\le b^2/N,$$

so some realisation attains the bound. That realisation is $\Phi\nu/\rho$ for
$\nu=\tfrac{\rho}{N}\sum_i\varepsilon_i\delta_{\theta_i}$, a sum of at most $N$ point masses of total
mass at most $\rho$, so $\nu\in\mathcal{F}_N^{\,\rho}$. The second display follows from
$\|\Phi\nu-y\|\le\|\Phi\mu-y\|+\|\Phi\mu-\Phi\nu\|$ applied at a near-minimising $\mu$. $\square$

The conclusion is about $\mathcal{F}_N^{\,\rho}$ and not about $\mathcal{F}_{N,\rho/N}$: two draws can
land on the same $\theta$, and the merged blob then carries more than $\rho/N$. The total mass is
what survives, which is the family Theorem 3's second identity is about.

The argument is the standard empirical-approximation one attributed to Maurey, and to Jones and
Barron; it is written out so that nothing is imported. Expanding the square with
$\inf_{B_\rho}J\le J(0)=\tfrac12\|y\|^2$ bounds the gap by $\rho b\|y\|/\sqrt N+\rho^2b^2/2N$.

The fixed budget is the hypothesis that limits it. At $\rho=NM$, the budget matching Theorem 3's own
family, the error term is $Mb\sqrt N$ and the theorem says nothing. At fixed $\rho$ it says the
ball's value is within $O(N^{-1/2})$ of the best $N$-blob encoding of that mass: blind to $N$ as a
constraint, not blind to $N$ in value.

Section 9 reports both regimes. Its second measurement lets the budget grow with $N$, which is
where the theorem is silent, so the two do not conflict. Its third holds $\rho$ fixed and finds the
gap closing exactly, at small $N$, by a cruder mechanism than the rate: the ball's own minimiser is
an $N$-blob encoding once $N$ reaches its support size. Theorem 8's bound stays orders of magnitude
above the measured gap throughout. The rate is correct and is never what decides the answer.

Corollary 7 assumed the objective is the error of a linearly rendered image. That hypothesis is not
a technicality. Drop it and convexification costs nothing at all.

> **Theorem 9 (a lift that loses nothing).** Write
> $\Lambda\mu=\big(\Phi\mu,\ (\Phi\mu)(\Phi\mu)^{\!\top}\big)\in\mathbb{R}^P\times\mathbb{S}^P$
> and $\ell(z,Z)=\tfrac12\|y\|^2-\langle y,z\rangle+\tfrac12\operatorname{tr}Z$. Then $\ell$ is
> affine, $\ell(\Lambda\mu)=J(\mu)$, and for every $N$ and $M$
> $$\inf_{\overline{\operatorname{conv}}\,\Lambda\mathcal{F}_{N,M}}\ell\ =\ \inf_{\mathcal{F}_{N,M}}J .$$

*Proof.* $\ell(\Lambda\mu)=\tfrac12\|y\|^2-\langle y,\Phi\mu\rangle+\tfrac12\|\Phi\mu\|^2=J(\mu)$.
An affine functional has the same infimum over a set as over its convex hull, because
$\ell(\sum_i\lambda_iu_i)=\sum_i\lambda_i\ell(u_i)\ge\min_i\ell(u_i)$, and the same infimum over the
closure by continuity. $\square$

The argument is trivial and is meant to be. What it contributes is not the lemma but the
identification: of Corollary 7's hypotheses, the one carrying the weight is the shape of the
objective, and this is what removing it costs and buys.

So the $N$-blob family has a convex relaxation with no gap, for every $N$. This does not contradict
Theorem 3. $\Lambda$ is quadratic in $\mu$, so it does not commute with $\operatorname{conv}$, and
the collision that collapses $\Phi\mathcal{F}_{N,M}$ leaves $\Lambda\mathcal{F}_{N,M}$ alone. Along
a convex combination,
$$\operatorname{tr}Z-\|z\|^2=\sum_i\lambda_i\|z_i\|^2-\Big\|\sum_i\lambda_iz_i\Big\|^2,$$
the variance the averaging introduced, and $\ell$ charges for it. Averaging blobs together is free
in measure space and costs something here.

The set has no description. Optimising $\ell$ over $\overline{\operatorname{conv}}\,
\Lambda\mathcal{F}_{N,M}$ is the original problem restated. The literature locates the difficulty
in the same place.

On a finite dictionary, cardinality-constrained least squares is reported to admit an **exact**
reformulation as a linear objective over the completely positive cone — a convex cone — with the
whole of the non-convexity moved into cone membership, which is NP-hard. So a convex formulation
that sees $N$ exists. It is Theorem 9's construction in finite dimensions, and it leaves
Corollary 7 by the same door: the objective is affine in the lifted variable rather than the error
of a linearly rendered image. So convexity by itself is not what obstructs. Corollary 7 needs both
halves — a convex feasible set and an objective of that form — and dropping the second removes the
gap and the description together. Both statements about the cone are from search summaries and were
not read.

---

## 6. Four ways out: three close, one changes the problem

**Limit each blob's amplitude.** This is the natural repair and it is what every mixed-integer
sparse solver does, under the name big-$M$. Theorem 1 kills it: blobs collide and the limit
exceeds the cap. Section 7 shows the cap does work when the parameter space is finite, and Section
9 measures what it is worth there.

**Change the topology.** Theorem 3 closes in the weak-\* sense, and the choice matters: in the
total-variation *norm*, $\mu_k\to\mu$ forces $\mu_k(\{\theta\})\to\mu(\{\theta\})$ at each point,
so the cap would survive and the norm-closed hull would be strictly smaller than the ball. One
could hope to build a relaxation there. Corollary 6 blocks it, because the collapse it states is
about the **un-closed** hull, where no topology appears. Whichever topology is preferred, the best
value a convex relaxation of the capped $N$-blob problem can return is the value of the plain
mass-ball problem. (Weak-\* is also the natural choice: bounded sets are weak-\* compact and are not
TV-compact, so minimising sequences have weak-\* limits and generally no TV limit.)

**Forbid blobs from colliding.** Restricting to encodings whose blobs are at least $\delta$ apart
blocks Theorem 1, and unlike the other three it is not undone by passing to the limit. Write

$$\mathcal{S}_\delta=\Big\{\sum_{i=1}^{N}a_i\delta_{\theta_i}\in\mathcal{F}_{N,M}:\
d(\theta_i,\theta_j)\ge\delta\ \text{ for } i\ne j\Big\}$$

with $\delta$ small enough that $N$ such blobs fit in $\Theta$, since otherwise $\mathcal{S}_\delta$
is empty and there is nothing to relax.

> **Theorem 10 (separation keeps the count).** Every $\mu\in\overline{\operatorname{conv}}\,
> \mathcal{S}_\delta$ satisfies $|\mu|(U)\le M$ for every open ball $U$ of radius $\delta/3$. Hence
> for $N\ge2$ the inclusion $\overline{\operatorname{conv}}\,\mathcal{S}_\delta\subseteq B_{NM}$ is
> strict, $NM\delta_\theta$ being excluded.

*Proof.* A ball of radius $\delta/3$ has diameter below $\delta$, so an element of
$\mathcal{S}_\delta$ places at most one blob in it and gives it mass at most $M$. Total variation is
subadditive, so $|\nu|(U)\le M$ for every $\nu\in\operatorname{conv}\mathcal{S}_\delta$. For $U$
open, $|\mu|(U)=\sup\{\int f\,d\mu:\ f\in C_c(U),\ |f|\le1\}$, where $C_c(U)$ is the continuous
functions vanishing outside a compact subset of $U$; each such integral is a limit of integrals
against elements of $\operatorname{conv}\mathcal{S}_\delta$, hence at most $M$.
$\square$

Theorem 10 is a possibility result. It shows the count can survive in the continuum, not that the
surviving structure can be used. So the separated hull lies in $B_{NM}$ under a cap on *local* mass,
and that depends on $N$ and $M$ separately: halve $N$ and double $M$ and $NM$ is unchanged while the
local cap doubles. Separation
restores in the continuum what Section 7 restores by discretising. That $\mathcal{S}_\delta$ is not
convex is no objection — neither is $\mathcal{F}_{N,M}$, and relaxation is the taking of a convex
hull.

The local-mass functional $\mu\mapsto\sup_\theta|\mu|(B(\theta,\delta/3))$ is convex, and weak-\*
lower semicontinuous as a supremum of such. Corollary 5 therefore forbids it from separating
$\mathcal{F}_{N,M}$ from $B_{NM}$, and it does not: $\mathcal{F}_{N,M}$ contains $N$ blobs stacked
in one ball. It sees $N$ only on the separated family — separation is a hypothesis on the
encodings, not a better penalty.

Three things are owed, and `experiments/e19_optimum_separation.py` settles the first. At the
exactly enumerated $N=3$ optimum on a $D=248$ dictionary of coherence 0.962, the chosen blobs are
not near-duplicates: their largest pairwise coherence runs 0.199–0.808 and their closest centres
3.8–25.5 pixels apart on a 48-pixel image. Some $\delta$ admits the optimum, so the separated
family is not empty of the answer. The same measurement shows where the route is likely to fail
instead. The cap $M$ must be at least the optimum's largest amplitude, which reaches 1.19 times
$\|y\|$ — the single-blob-carries-the-image effect the fourth measurement in Section 9 reports —
and the mass budget $NM$ that implies is 1.45–2.27 times the mass the optimum actually spends,
which is where the second measurement finds the ball already saying nothing. The local cap would
have to carry the argument by itself.

The second and third are settled too, and against the theorem.
`experiments/e20_local_mass.py` computes the programme rather than speculating about it. On a
finite dictionary the local cap is one linear constraint per ball centre, so the relaxation is a
quadratic program, and it nests exactly: every ball contains its own centre, so the local
constraints already imply $\|c\|_\infty\le M$ and the set sits inside Theorem 11's polytope
$P_{N,M}$, which in turn sits inside the mass ball. The three bounds are therefore ordered, and the
difference between the second and the third is what Theorem 10 contributes and nothing else.

Both parameters are read off the enumerated optimum — $M$ its largest amplitude, $\delta$ its
smallest pairwise separation — which is the tightest admissible pair, since anything smaller in $M$
or larger in $\delta$ excludes the answer. The radius used is just under $\delta/2$ rather than the
$\delta/3$ of the theorem's statement: the proof needs only that a ball's diameter stays below
$\delta$, and the wider ball holds more atoms and binds harder. This is the theorem's best case.

Measured as the share of the distance from the mass-ball bound to the truth, over six images at
$D=248$, $N=3$: big-$M$ recovers **0.0000** of it, on every image. Theorem 10's set recovers at
most **0.0155** — about one and a half per cent — and nothing at all under the coherence metric.
The balls are not the problem; they hold 23 atoms on average. The amplitudes are. The mass-ball
solution's largest single amplitude is already at most $0.745M$, so the per-blob cap is inactive
before it is imposed, and its largest ball mass only just crosses the local cap. A cap set by an
optimum whose largest blob carries the energy of the image is too large to constrain anything, and
making it local does not change that.

So Theorem 10 is true, its programme is tractable, its set is strictly smaller in three of the six
instances — and it is worth about one per cent. The one remaining gap is that all of this is on a
grid: the continuum statement is not reachable this way, and how much smaller the continuum hull is
has still not been computed.

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

> **Theorem 11 (a finite dictionary keeps the count).** Let $1\le N<D$ with $N$ an integer, and
> $M>0$. Then
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
Theorem 11. Spaces that mix the two are not treated here and do not arise in this work.

The blob count is representable convexly when the parameter space keeps blobs apart, and Theorem 10
gets the same from keeping the *encodings* apart on a space that would not. What decides it either
way is whether two blobs can be driven together, not which penalty, dictionary or image is in play.
A finite dictionary is the cheap way to obtain that and the one measured here, not the only one.

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

Eight results from this repository, each described in enough detail to be read without the other
document. The first three concern the mass-ball relaxation, the next three the amplitude caps that
a finite dictionary allows, and the last two the dictionary's coherence and the certificate.

Every number below comes from images of 32 to 64 pixels a side and dictionaries of 59 to 768 blobs,
at $N\le4$ where the optimum is enumerated exactly and $N\le64$ where it is searched. That is three
orders of magnitude below the budgets an encoder uses. The theory does not depend on scale; none of
these figures should be quoted as though it does.

**The gap is real with both problems solved exactly.** `experiments/e4_exact_l0.py`. Shrink the
problem until neither side can be blamed on a solver: a $32\times32$ image, a dictionary of $D=768$
blobs, and $N\le4$. Solve $\ell_0$ by enumerating every one of the $\binom{768}{3}=75{,}202{,}816$
supports, and $\ell_1$ by following its exact solution path as $\lambda$ decreases — the path is
piecewise linear, so it can be computed without approximation — then re-fitting amplitudes on the
blobs it selects, which removes the shrinkage bias and can only help it. At matched blob count
$\ell_1$ is
worse by between **6% and 409%** across four targets and budgets from 2 to 4 blobs. The two agree
only at $N=1$, where the problems coincide. The worst case, 409%, is on a target built as an exact
sum of three dictionary blobs, at $N=2$: $\ell_0$ reaches error 8.70 and $\ell_1$ 44.25. On a
cartoon target at $N=2$ the figures are 18.00 and 30.90, and the two answers share no blob at all.
One qualification the experiment does not wear on its face: the enumeration is restricted to
supports whose refitted mass is within twice the larger of greedy's and $\ell_1$'s, a guard against
near-duplicate pairs with huge cancelling amplitudes. So the $\ell_0$ side is the exact optimum of
a mass-capped problem, not of the unconstrained one. The direction is safe — dropping the guard can
only lower the $\ell_0$ error and widen the gap — but "solved exactly" means exactly this problem.

Theorem 3 says no better convex penalty removes the gap; the experiment says how large it is when
nothing is approximate.

**A mass-constrained bound goes vacuous quickly.** `experiments/e3_absolute_bound.py`. Theorem 3
says a total-mass constraint is the most any measure-space convexification can see, so a lower
bound on (P0) built from one is the best of its kind. Measured on $64\times64$ images against
budgets 1 to 16, it stops saying anything by $N=4$–8, where it falls to zero. At $N=1$, where the
single best blob can be found exactly by direct search so the true optimum is known, the bound
reaches 66–76% of it and leaves 24–34% it cannot exclude.

**At a fixed mass budget the same relaxation is exactly tight, at small $N$.**
`experiments/e16_fixed_mass.py`. The previous measurement lets the budget grow with $N$, which is
the regime Theorem 8 says nothing about. Holding $\rho$ fixed and sweeping $N$ instead, on the
$D=768$ dictionary at $32\times32$: the gap between the ball and the best $N$-blob encoding of the
same mass closes **exactly** — not asymptotically — at $N_0$ between **3 and 48** across twelve
target-and-budget cells, and at 4 to 12 for the two smaller budgets. On the cartoon target at
$\rho=\|y\|$ the certified gap runs 17.3%, 3.5%, 1.0%, 0.3%, 0.007% of $\tfrac12\|y\|^2$ at
$N=1,2,3,4,6$ and is zero by $N=12$. No rate is being confirmed. The ball's own minimiser is a
feasible $N$-blob encoding once $N$ reaches its support size, so the gap is zero from there on, by
a mechanism cruder than the theorem's. Theorem 8's bound over the same range reads 300%, 191%,
149%, 125%, 98% — it does not fall below the trivial
$\tfrac12\|y\|^2$ until $N=4$–24, and at that budget the measured gap is already under 0.13% of
$\tfrac12\|y\|^2$ in all twelve cells. The rate holds and is never what determines the answer.

This is not a rehabilitation of the bound. The `free mass` column records what the *uncapped* best
$N$-blob encoding wants. At $\rho=0.75\|y\|$ on the cartoon, where $\rho$ is 10.6, it reads 11.5,
15.8, 17.1, 17.2, 26.4, 32.1 over $N=1$ to 8 — above the cap at every budget — and beyond that it
becomes erratic (135, 96, 637, 6276) as the unconstrained fit starts cancelling large opposing
amplitudes on a coherent dictionary. Only the small-$N$ end of that column carries information, and
there it says the cap binds everywhere, so every row with a small gap is a row where the problem is
not (P0). Where the cap does not bind — the larger budgets, closest to the encoding problem — the
gap at $N=1$ is 30.6% rather than 7.3% and $N_0$ is 48 rather than 8. Across the sweep, tightness
and relevance move in opposite directions.

A note on method. On the in-model target at $\rho=2\|y\|$ the ball reaches error zero, its minimiser
is not unique, and a first-order method returns a dense one with 730 non-zeros. The sparsest
ball-optimal encoding has three, which $N_0$ recovers. Reporting the returned minimiser's support
size would have recorded 730, which is why the number above is the measured $N_0$.

**The per-blob cap survives discretisation and is still useless here.**
`experiments/e8_branch_and_bound.py`. On a finite dictionary Theorem 11 applies, so branch-and-bound
has something to work with. Implemented on a $48\times48$ image with $D=248$: the node bound is
informative only when a certain ratio falls below 1, and at the tightest cap the search may legally
use — $M$ set to the largest amplitude in the best solution found so far, below which that solution
would itself be excluded — the ratio is **5.7 at $N=4$ and 8.9 at $N=6$**. Two hundred thousand
nodes returned a 100% gap. The reason is specific to Gaussian blobs: the incumbent's largest
amplitude is 24.0 while
$\|y\|=21.2$, so **a single blob carries the energy of the whole image**, and any cap large enough
to admit it is far too large to constrain $N$ blobs. This is a different failure from the
continuum one. There the cap is destroyed by collision; here it survives and is merely far too
generous.

**The perspective relaxation fails the same way.** `experiments/e9_perspective.py`. The standard
strengthening from the sparse-regression literature, applied at $D=248$, $N=3$ on $48\times48$
images. It only applies to a modified problem carrying an extra penalty $\lambda_2\|c\|^2$ on the
amplitudes. Where that penalty is small enough to leave the problem essentially unchanged
($\lambda_2\le10^{-3}$, where the reconstruction moves from 7.6923% to 7.6931% on one target and
19.4843% to 19.4877% on another) the gap between the bound and
the true optimum, before any branching has happened, is **64–86%**. It closes only at
$\lambda_2=1$, where the reconstruction error has risen by a factor of 2.4–4.3. There is no setting in which the bound is
tight and the problem is still the one wanted. Two relaxations failing for one shared reason —
amplitudes at signal scale — points at the dictionary rather than at the choice of relaxation.

**How much a perspective-type relaxation has to work with is computable.**
`experiments/e17_separable_mass.py`. Such a relaxation acts on the part of the quadratic that is a
*separable* function of the coefficients: it splits $c^{\!\top}(A^{\!\top}\!A+\lambda_2I)c$ into
$c^{\!\top}Dc$ with $D$ diagonal and a convex remainder, then replaces $c_j^2$ by $c_j^2/z_j$. That
separable piece is how it escapes Corollary 7, and its size follows from the eigendecomposition.
The largest admissible $\operatorname{tr}D$ here, certified:

| $\lambda_2$ | 0 | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ | 1 |
|---|---|---|---|---|---|
| separable share of the quadratic | **0.15%** | 1.30% | 3.95% | 15.27% | **57.70%** |

The cause is rank deficiency: the Gram has rank 531 of 768, and all 768 coordinates carry energy in
the null space, so an exact null vector forces $d_j=0$ at every coordinate. At $\lambda_2=0$ there
is nothing to strengthen, so the 64–86% is not an artefact of the implementation and a better one
would not remove it. The share reaches half only at $\lambda_2\approx1$, where $D=\lambda_2I$ alone
supplies it, and where the reconstruction is 2.4–4.3× worse. Raw material and damage arrive
together. The sparse-regression literature reports the same dependence as a diagonal-dominance
condition at $\lambda_2=0$; from search summaries, not read.

**On a grid, the relaxation is tight only where the dictionary is useless.**
`experiments/e4_exact_l0.py`, output `results/e4_coherence.txt`. Theorem 11 restores the dependence
on $N$ but says nothing about how *tight* the resulting relaxation is. Pruning the dictionary by
greedy decorrelation and re-solving both problems exactly at $N=3$ on a $32\times32$ image:

| coherence | $D$ | exact $\ell_0$ error | $\ell_1$ excess | shared support |
|---|---|---|---|---|
| 0.985 | 768 | 11.86% | **+145%** | 1 of 3 |
| 0.900 | 470 | 15.23% | +91% | 1 of 3 |
| 0.750 | 103 | 22.20% | +19% | 2 of 3 |
| 0.599 | 73 | 33.16% | +29% | 1 of 3 |
| 0.448 | 59 | **46.61%** | +5% | 2 of 3 |

Coherence is the largest inner product between two normalised dictionary blobs; $\ell_1$ theory
wants it well below 1 and here it starts at 0.985. Decorrelating the dictionary destroys the
dictionary's ability to approximate: the error at coherence 0.448 is four times the error at 0.985.
Near-orthogonal Gaussians cannot represent an image in few blobs.

The gap column is not monotone, and an earlier version of this table omitted the row that shows it
— the 0.599 row is worse than the 0.750 row above it. The same sweep in the same output file runs
on two further targets and is less orderly still: on `ascent` the excess falls to zero and stays
there, on `face` it runs 81%, 62%, 2%, 43%, 8%. What survives is the comparison of the endpoints,
which is large and holds on all three targets: the relaxation becomes tight in the regime where
there is nothing worth approximating. The claim that it does so *monotonically* is not supported.
Both sides here are the mass-capped $\ell_0$ described above.

**The certificate's value does not indicate (P0) optimality either.**
`experiments/e14_certifiable.py`. Section 8 says the certificate cannot *prove* (P0) optimality.
Its numerical value might still correlate with being optimal and serve as a heuristic, which is a
separate question. Across 178 distinct local optima on 12 images at $N=3$, the certificate value at
$\lambda=0$ separates the best solution from the rest with AUC **0.560**, 95% bootstrap interval
over images **[0.467, 0.666]** — the probability
that it ranks the best solution ahead of a randomly chosen worse one, where 0.5 is chance. The
interval contains chance, so on this population the experiment does not distinguish the certificate
from a coin. An information-free control drawn from a random number generator reads 0.363,
[0.234, 0.523]; with 12 images the intervals are too wide for either to be separable from 0.5.

Two qualifications. The verdict is population-specific: in a leg where the optimum is known by
enumeration and the comparison is against the 200 lowest-error supports, the same quantity reads
0.650, [0.578, 0.724], above chance. Those supports are mostly not local optima, so that leg asks a
different question from the one a restart-based procedure faces. And the experiment maximises the
certificate over a 248-blob dictionary rather than over all of $\Theta$, so it measures a lower
bound on the true certificate value.

---

## 10. Limits

The proofs are self-contained, so what is exposed is the hypotheses and the measurements.

**The theorems need blobs to be able to collide.** Theorem 1 needs points of $\Theta$ arbitrarily
close to one another, and Theorem 3 needs that everywhere. A parameter space of isolated points is
the finite-dictionary case of Section 7, where the conclusion reverses; a parameter space that is
continuous but whose *encodings* are required to be separated is Theorem 10, where it also reverses.
Every claim here about the continuum is a claim about a parameter space in which blobs can be
placed arbitrarily close together, and about encodings free to do so. That is what continuous
placement means, but it is an assumption and is stated as one.

**They need $\varphi$ continuous on $\Theta$.** Corollary 6 rests on the weak-\* continuity of $J$,
which needs $\theta\mapsto\varphi_\theta$ continuous. Widths bounded away from zero give this. At
zero width the blob degenerates and continuity is lost — a different failure, not covered here.

**A concrete refutation would be** a convex, weak-\* lower semicontinuous $R$ and a level $c$ with
$R\le c$ on $\mathcal{F}_{N,M}$ and $R>c$ somewhere on $B_{NM}$, for a compact $\Theta$ with no
isolated points. Corollary 5 says this cannot exist, so exhibiting one would locate an error in
Lemma 2 or Theorem 1.

**The measurements are less secure than the proofs.** They are on Gaussian dictionaries of a few
dozen to a few hundred blobs, on images of 32 to 64 pixels a side, at $N\le4$ where the optimum is
enumerated and $N\le64$ where it is searched. The theory says the gap cannot be removed by a better
convex penalty; it does not predict the gap's *size*, and the sizes reported are properties of
Gaussian blobs on small images, not universal constants. In particular "a single blob carries the
energy of the whole image" is a fact about this dictionary and would not hold for, say, a wavelet
frame with uniformly bounded atom amplitudes.

**Not claimed: that the blob count is the right budget.** Section 1 takes it as given, because
(P0) does. Amplitudes here reach the scale of the image norm, so under a real quantiser their
precision costs bits that a count does not measure. Every result below the budget line — what the
relaxation loses, and what that loss is worth — is stated in blobs. Whether it survives being
restated in bits is untested, and it is the assumption most likely to change the picture.

**Not claimed: that (P0) is unsolvable.** The result is about convexification. Direct methods are
untouched, and on a finite dictionary at $N=3$, local search from 100 random starts reached the
exhaustively verified optimum on 40 of 40 images.

**Not claimed: that the mass-ball relaxation is worthless.** It is a valid lower bound on (P0), and
Section 9 measures it as exactly tight from a small $N$ onwards at a budget held fixed. Its limit is
not looseness but the budget. (P0) fixes the blob count and says nothing about mass, so choosing
$\rho$ needs the optimum's mass, which is not available.

**Not claimed: that no convex formulation exists anywhere.** Corollary 7 needs two things at once,
a convex feasible set and an objective that is the error of a linearly rendered image. Section 5.2
gives up the second and the gap disappears. Three things therefore sit outside Corollary 7: a lift
whose objective is not a function of the rendered image alone, as the perspective relaxation of
Section 9 is not; a lift with a non-convex constraint, which is where the blob count lives in a
moment hierarchy; and the finite dictionary of Section 7. The moment work located in searching
applies the hierarchy to the dual constraint $|\eta|\le1$ for tractability rather than to the count;
it was not read.

---

## 11. Status of each claim

| claim | status |
|---|---|
| Theorems 1, 3, 4, 8, 9, 10, 11; Lemma 2; Corollaries 5, 6, 7 | proved above; nothing imported |
| Extreme points of the total-variation ball are the signed point masses | standard; Lemma 2 proves what is used, so it is not relied on |
| $\overline{\operatorname{conv}}\,\mathcal{F}_N^{\,\tau}=B_\tau$ (second half of Theorem 3) | proved above, but not new: it is the definition of an atomic-norm ball |
| Theorem 11 is the convex hull of the big-$M$ sparse set | not new; $\ell_1$ is the convex envelope of $\ell_0$ on the $\ell_\infty$ ball. A search points to Kim, Tawarmalani & Richard, *Convexification of Permutation-Invariant Sets*, for a general treatment; what exactly it covers is from a search summary and the paper was not read. The proof above is independent of both |
| An $\ell^2$ cap gives a $k$-dependent hull in the finite setting | proved above, by the $k=1$ and $k=D$ cases. It is the ball of the $k$-support norm, which is known, but nothing here needs that |
| Some optimum of the mass-constrained problem uses $\le P+1$ blobs | proved above from Carathéodory. Representer theorems give sharper versions; none is used |
| Maurey / Jones / Barron attribution for Theorem 8 | attribution from search summaries, never read; the proof is given above |
| BLASSO formulation and its dual certificate | from `papers/1811.06416v1.pdf` |
| Global convergence of Conic Particle Gradient Descent for (P$\lambda$) | statement from `papers/1907.10300v2.pdf`; its irrelevance here is the argument of Section 8 |
| $\ell_1$ versus $\ell_0$ at matched $N$, both exact | measured, `experiments/e4_exact_l0.py` |
| Mass-constrained bound goes vacuous by $N=4$–8 | measured, `experiments/e3_absolute_bound.py` |
| Big-$M$ node bound off by 5.7–8.9× | measured, `experiments/e8_branch_and_bound.py` |
| Perspective relaxation root gap 64–86% | measured, `experiments/e9_perspective.py` |
| Separable share of the quadratic is 0.15% at $\lambda_2=0$, 57.7% at $\lambda_2=1$ | computed, `experiments/e17_separable_mass.py`; a certified upper bound on the raw material, not on the resulting gap |
| Theorem 9 | proved above |
| Coherence governs tightness and costs approximation power | measured, `results/e4_coherence.txt`, at the endpoints only; the sweep is not monotone and is less so on two of its three targets |
| Big-$M$ node bound figures (5.7–8.9×, the 24.0 amplitude, 200,000 nodes) | **the raw output of `e8_branch_and_bound.py` is not committed.** The numbers are quoted from a run that left no artifact in this repository, which by the standard applied everywhere else here is not good enough. e19 corroborates the amplitude scale independently, on a different enumeration; the rest is uncorroborated until the run is repeated and its output committed |
| Certificate value not separable from chance on the restart population | measured, `experiments/e14_certifiable.py`; the bootstrap interval is the claim, not the point estimate |
| Branch-and-bound solvers reaching $10^7$ variables | from search summaries; the sources were never read |
| Moment hierarchies for the BLASSO target the dual constraint, not the count | from search summaries; not read, not tested |
| Optima are separated, but the cap they force leaves the mass budget 1.45–2.27× slack | measured, `experiments/e19_optimum_separation.py` |
| Theorem 10's programme is a QP on a grid, and recovers $\le1.6\%$ of the distance from the mass ball to the truth | measured, `experiments/e20_local_mass.py`, at the tightest admissible $\delta$ and $M$ and the widest valid ball radius. Big-$M$ recovers 0.0% on the same instances |
| How much smaller the *continuum* separated hull is | open; e20 measures only the grid restriction |
| Prior art for Theorem 10's construction | searching turned up the separation hypothesis only in its recovery role, not as a way of convexifying the count. Per M6 of the companion document a null search result is not novelty, and this row records a failed search, not a claim of priority |
| Gap closes exactly at $N_0=3$–48 at fixed mass; Theorem 8's bound never binding | measured, `experiments/e16_fixed_mass.py`. $U$ is a search upper bound above $N=2$, so the gap is an upper bound; small values are conclusive, large ones may be the solver |
| Theorem 8's $N^{-1/2}$ rate itself | not isolated. The measured decay is dominated by the support-size mechanism, so these runs do not test the rate |
