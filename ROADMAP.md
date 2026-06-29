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
  per-generation in-kernel variant under `de_sync = TRUE` (0.4.1) for the
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
- 0.4.0 (target 2026-06-29): Differential Evolution MCMC for correlated
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
  synthesis path of 0.8.0 introduces.
- 0.4.1: the per-generation in-kernel Differential Evolution path under
  `de_sync = TRUE`, for the canonical per-iteration mixing on curved or
  strongly correlated targets.
- 0.5.0: bivariate copula workflow with the four common families.
- 0.6.0: per-column marginal auto-selection.
- 0.7.0: vine copula for `d > 2`.
- 0.8.0: synthesis, `generate(fit, n)`.
- 1.0.0: documentation and API polish as the reference release.

Tier 2 (fused kernels for common likelihoods) and Tier 5 (vector and matrix
DSL) become optional, scheduled only when a real user case pulls them.

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
through v0.8.0 march already on the calendar.
