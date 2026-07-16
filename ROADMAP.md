# gpumetropolis roadmap

This file records the direction beyond the first release. It is a roadmap,
not a commitment: the priority was to ship a focused v0.x first. The items
below are tiered by how they relate to the current scope and to the binding
decisions of `~/.claude/plans/gpumetropolis-sprint.md`.

## North star

The longer arc is a portable probabilistic computing runtime: a generic model
DSL whose compiled representation runs across CPU and GPU vendors from one
source. The current package, a generic vendor-agnostic Metropolis-Hastings
sampler, is the foundation of that arc. The arc is multi-year; it is not the
plan for v0.x.

## Tier 1: in scope, natural next steps

These stay inside the Metropolis-Hastings family (their accept step is still
Metropolis-Hastings), so they do not reopen decision #1, and they fit the
many-chains GPU architecture.

- Adaptive Metropolis (delivered, 0.2.0): adapt the proposal scale per chain
  during warmup through Welford and Robbins-Monro, so `proposal_sd` is a seed
  rather than a knob the user tunes by hand.
- Parallel tempering (delivered, 0.3.0): chains at several temperatures with
  swaps, for multimodal targets. The tempered chains parallelise across the
  chain axis.
- Differential Evolution MCMC (0.4.0): a population sampler whose proposals
  use the differences of other chains, so the proposal aligns with the
  correlation of the target through the ensemble geometry, with no explicit
  covariance. Population-based, so it maps onto the chain axis the package
  already parallelises. Two paths share `method = "de"`: a host-orchestrated
  variant with a per-batch frozen population snapshot (0.4.0, default), and a
  per-generation in-kernel variant under `de_sync = TRUE` (0.4.2) for the
  canonical per-iteration mixing on harder targets.

## Tier 2: optimisation layers

- Fused kernels for common likelihoods: hand-written CubeCL kernels for the
  normal, logistic and Poisson likelihoods, faster than the generic bytecode
  interpreter, used as fast paths beside the DSL.
- Mixed precision: an f16 data path with f32 accumulation, with the automatic
  delayed-acceptance bias correction already recorded in the canonical plan.

## Tier 3: needs an explicit decision

- Optional automatic differentiation: differentiating the DSL expression is
  tractable because the operation set is small. But its payoff is gradient
  samplers (HMC, NUTS, MALA), which are outside the Metropolis-Hastings family
  and reopen decision #1 of the plan. Only with an explicit instruction from
  the author.

## Tier 4: weak fit, recorded honestly

- Slice sampling on the GPU: slice sampling does a variable number of
  density evaluations per step (stepping-out, shrinkage), which causes warp
  divergence. It is, against intuition, the listed item that fits a GPU
  worst. It would be a CPU-side feature at best.

## Tier 5: separate paradigm, large scope

- Sequential Monte Carlo backend: a different paradigm; its resampling step is
  a global communication, harder on the GPU.
- Vector and matrix support in the DSL: a substantial expansion of the DSL,
  the bytecode VM and the kernel, but the path to regression and hierarchical
  models.

## Tier 6: premature

- Sparse tensor support: deferred until a concrete target model needs it.

## Application direction: copula inference and synthetic data

The application direction is now part of the package itself, not a sibling
project. The package will grow from a generic Metropolis-Hastings sampler
into an MCMC-driven copula synthesis engine, all under the `gpumetropolis`
name. Decision recorded 2026-06-02.

The engine is already close to one useful slice: Bayesian inference of the
parameter of an extreme-value copula, the Gumbel family, from pseudo-
observations. The Gumbel copula density is inside the DSL operation set, and
the DSL already accepts several data columns. A bivariate demonstration is a
vignette away and needs no engine change.

The path forward, in dimension order:

- Bivariate copula workflow (`gpum_copula(data, family)`): Gumbel, Clayton,
  Frank and Gaussian families on two-column data. In reach with the current
  scalar DSL.
- Per-column marginal auto-selection: type detection on a data.frame,
  candidate-family search per column, MCMC fit per marginal. Each column
  produces a fitted CDF.
- Vine copula for `d > 2`: pair-copula decomposition (Aas, Czado, Frigessi
  and Bakken 2009) with structure selection (Dissmann, Brechmann, Czado and
  Kurowicka 2013). The vine decomposes a `d`-dimensional dependence into
  `d(d-1)/2` bivariate copulas, each tractable in the current DSL. This is
  the path that sidesteps the full vector DSL of Tier 5.
- Synthesis: `generate(fit, n)` samples from the fitted vine, inverts each
  marginal CDF and returns the synthetic data.frame.

The DSL expansion of Tier 5 leaves the critical path of the application:
vines reduce dependence to pairs, and pairs fit in the current DSL. Tier 5
is revisited only if a user reports a case that vines and bivariate copulas
cannot serve.

Recorded boundary: a neural generative model, a variational autoencoder or a
normalising-flow copula, is a different computational paradigm, gradient
training and automatic differentiation. It is not built on this engine and is
not part of this package's arc. If that route is wanted it is a separate
project with a separate stack.

## Sequencing

The release trajectory is the spine of the roadmap; each release is a
discrete deliverable, validated before the next opens.

- 0.2.0 (delivered 2026-06-02): Adaptive Metropolis diagonal per chain,
  per-chain proposal_sd in the kernel, `backend = "auto"` as default and
  `gpum_diagnose()` as the one-call diagnostic.
- 0.3.0 (delivered 2026-06-15): Parallel Tempering for multimodal targets.
- 0.4.0 (delivered 2026-06-29): Differential Evolution MCMC for correlated
  targets without manual covariance, host-orchestrated with a per-batch
  population snapshot. Ships `gpum_crlb()`, an optional Cramer-Rao reference
  that compares the posterior spread to the inverse observed Fisher
  information on regular targets, with guards that withhold the comparison
  where its assumptions fail. Ships a formal Bayesian decision and comparison
  layer: posterior probability of a hypothesis and the ROPE-and-HDI rule from
  the draws, WAIC and PSIS-LOO for predictive comparison, and the marginal
  likelihood and Bayes factor by thermodynamic integration. The posterior
  predictive check and Bayesian p-value are deferred to a later release, since
  they need replicated-data generation from an arbitrary likelihood, which the
  synthesis path of 0.9.0 introduces.
- 0.4.1 (delivered 2026-06-29): a plotting layer for the joint posterior
  (pairs, credible regions, level-curve and 3D surface), explanatory figures
  for the Bayesian decisions, the observed-against-generated check
  (`gpum_ppc`, `gpum_density_compare`), and three real-data case-study
  vignettes with a live head-to-head against the established packages.
- 0.4.2 (delivered 2026-07-03): the synchronous per-generation Differential
  Evolution path under
  `de_sync = TRUE`, for the canonical per-iteration mixing on curved or
  strongly correlated targets.
- 0.5.0 (delivered 2026-07-16): extreme optimisation for honest competitive
  performance, pulled ahead
  of the application arc because the didactic book is gated on it. The live
  head-to-heads of the 0.4.1 case studies showed the package competitive but
  not
  ahead on small single-chain problems, where a tuned conjugate or gradient
  sampler leads. This milestone closed that gap by engineering the adaptive
  warmup: full-covariance proposals from a cross-chain pooled covariance,
  dimension-dependent acceptance targets, a mid-warmup restart of the
  accumulators and the Robbins-Monro schedule, and an early-stopping
  `warmup = "auto"`. The outcome is recorded as amendment v1.1 of
  `EXPERIMENT_PROTOCOL.md`: the `rwm` path wins all three applied cases
  against Stan and the generic samplers, median of three runs on the
  versioned harness. The didactic book below is gated on this
  milestone, so its claim that the package is the right tool is earned
  experimentally.
- 0.5.1 (delivered 2026-07-16): the conjugate fast path. `gpum_lm()` declares
  the Gaussian linear model under a proper Normal-inverse-Gamma prior and the
  fit samples the closed-form joint posterior exactly, independent draws with
  the effective sample size equal to the draw count, plus the closed-form log
  marginal likelihood. Closes the regime v1.1 conceded to the Gibbs
  specialists at its root: measured at 6.4 and 7.3 million effective draws
  per second on the conjugate regressions, against 1.2 and 0.72 million for
  `MCMCregress`.
- 0.5.2 (delivered 2026-07-16): the gradient. Reverse-mode automatic
  differentiation of the model bytecode, JIT-compiled to native code, drives
  `method = "mala"` (Metropolis-adjusted Langevin with MALTA drift
  truncation, preconditioned by the 0.5.0 pooled covariance) and the exact
  observed-information path of `gpum_crlb()`. Closes the regime v1.1
  conceded to gradient samplers: on the d = 21 logistic regression `mala`
  reaches 30262 effective draws per second against Stan's 9966, and it
  multiplies the applied-case wins (amendment v1.2 of
  `EXPERIMENT_PROTOCOL.md`). CPU backend in this release; the GPU port of
  the gradient kernel is future work.
- 0.5.3 (delivered 2026-07-16): inference beyond i.i.d. data by the
  conditional-factorisation principle. `gpum_ts_model()` assembles the
  lagged design of a Markov model of any finite order, with exogenous
  covariates for the fully non-i.i.d. case; `gpum_lfo()` gives exact
  leave-future-out cross-validation where WAIC and LOO lose their
  exchangeability license. The documentation maps the asymptotic theory
  per regime (Lindeberg-Feller and LAN for independent non-identical
  rows; martingale CLT, Billingsley 1961 and the Markov Bernstein-von
  Mises of Borwanker, Kallianpur and Prakasa Rao 1971 for dependence),
  and the `case_nile_timeseries` vignette exercises the whole layer on
  the Nile flow with the 1898 level shift. Declared boundary: latent
  recursions and matrix-coupled likelihoods wait for the matrix-DSL tier.
- 0.6.0: bivariate copula workflow with the four common families.
- 0.7.0: per-column marginal auto-selection.
- 0.8.0: vine copula for `d > 2`.
- 0.9.0: synthesis, `generate(fit, n)`.
- 1.0.0: documentation and API polish as the reference release.

Tier 2 (fused kernels for common likelihoods) and Tier 5 (vector and matrix
DSL) become optional, scheduled only when a real user case pulls them.

## The didactic book on Bayesian inference

The headline deliverable of the arc is a didactic book on Bayesian inference,
written to an academic standard for use as a university reference, in which
`gpumetropolis` is the working tool throughout. Provisional title, in English:
**From A Priori to A Posteriori**, with the subtitle *a rigorous, hands-on
course in Bayesian inference, with the classical view alongside*. The Latin
terms name the arc: from reasoning before the data, derived by hand, to the
posterior, computed. The book teaches the inference,
declaring the model, sampling, diagnosing convergence, deciding hypotheses,
comparing models, checking the fit against the data, and uses the package at
every step as the tool that fits the task. It is not a manual for the package;
it is a course in the method, with the package as its instrument.

Three differentiators set its character. First, mathematical rigor is kept
throughout: results are derived and proved, not asserted, to the standard of a
graduate text. Second, a frank, constructive comparison with classical
inference runs alongside the Bayesian development, taking Casella and Berger as
the reference for the classical side, so the reader sees the two paradigms
answer the same question and meet or part on stated terms, step by step rather
than by decree. Third, the pedagogy is hands-first: the reader learns the
mathematics and works each case by hand on paper, derives the maximum
likelihood estimator and the Fisher information, then reproduces the numbers
with the classical tools and with the Bayesian sampler and compares them. The
seed is already in the case studies, where the Cramer-Rao bound from
`gpum_crlb` is shown matching the standard errors of `lm` and `evd::fgev`, and
the posterior mean tracking the maximum likelihood estimate; the book makes
that meeting of paradigms the spine of the teaching.

The comparison extends to machine learning as unifying mathematics, not as a
demonstration of the package on deep models. The book draws the exact bridges
that a reader from machine learning already half-knows: the prior is a
regulariser, so the posterior mode under a Gaussian prior is the L2-penalised
maximum likelihood and under a Laplace prior the L1; a loss is a negative
log-likelihood, so cross-entropy and squared error are the categorical and
Gaussian likelihoods; and WAIC and LOO are the principled counterpart of
cross-validation and the control of overfitting. It stops at the boundary
rather than crossing it: full Bayesian neural networks, variational
autoencoders and normalising flows are high-dimensional differentiable targets,
the territory of gradient samplers and automatic differentiation, which is not
this package's arc (decision 6 of `BRIEFING.md`). Where the inference needs
that regime, the book names the right tool, gradient methods through Stan or
the JAX-backed libraries, and says plainly why a generic sampler does not go
there. Three paradigms, classical, Bayesian and machine learning, answer the
same question and meet or part on stated terms.

Two conditions govern it. First, sequencing: the book comes only after the
0.5.0 extreme-optimisation milestone, so that its claim that `gpumetropolis` is
the right tool is earned in fair, honest, experimental head-to-heads against
the established packages, not asserted. A textbook that recommends a tool must
show that tool winning on the merits. Second, direction: the book guides the
package, not the other way round. Each chapter that the teaching needs surfaces
a feature the package must deliver well, and those needs set the priority of
the improvements before it. The living-book vignette chapters already written
are the seed; the inference and comparison theory developed alongside them, and
the registered benchmark, are the rest.

Form: a bookdown volume in the style of an R package book, alongside a pkgdown
reference site for the function documentation, published on the web and
deposited for a DOI.

## Post-1.0.0 trajectory

Through 1.0.0 the identity of the package is the deliberate one of
decisions 1 to 9 in `BRIEFING.md`: a portable Metropolis-Hastings sampler
that becomes a copula synthesis engine. The package is intentionally not
competing on the algorithmic state of the art over that window. The
positioning section of the README states the trade explicitly: Stan,
PyMC and NumPyro dominate differentiable unimodal targets through
HMC/NUTS, and `gpumetropolis` makes no claim against them in that regime.

After 1.0.0 the orientation shifts. The post-1.0.0 trajectory aims at
genuine state-of-the-art status in the narrower intersection that the
package occupies: **MCMC plus vendor-agnostic GPU portability plus
copula-driven synthesis**. Two candidate directions, to be decided when
the 1.0.0 release lands:

- **HMC and NUTS** as a parallel sampler in the package, which would
  reopen decision 3 of `BRIEFING.md` (automatic differentiation). The
  cost is the AD machinery on the bytecode and the kernel; the benefit
  is matching `Stan` on differentiable targets while keeping the
  many-chains GPU axis.
- **Hold the no-AD line and double down on the niche**: make the copula
  synthesis path through vines and parallel tempering the best
  implementation in any language, on GPU, with explicit support for
  large `d` through vine decomposition and large `n` through the
  data-parallel kernel. The state of the art here is currently
  scattered across small specialised packages; a single coherent path
  has room.

The concrete choice between these two directions is recorded as a
post-1.0.0 decision, after 2026-11-02. The interim is the v0.4.0
through v0.9.0 march already on the calendar.
