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

- Adaptive Metropolis: adapt the proposal covariance during warmup. Removes
  the burden of tuning `proposal_sd` by hand, the main usability gap today.
- Differential Evolution MCMC: a population sampler whose proposals use the
  differences of other chains. Population-based, so it maps onto the chain
  axis the package already parallelises.
- Parallel tempering: chains at several temperatures with swaps, for
  multimodal targets. The tempered chains parallelise across the chain axis.

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
- 0.3.0 (target 2026-06-15): Parallel Tempering for multimodal targets.
- 0.4.0 (target 2026-06-29): Differential Evolution MCMC for correlated
  targets without manual covariance.
- 0.5.0: bivariate copula workflow with the four common families.
- 0.6.0: per-column marginal auto-selection.
- 0.7.0: vine copula for `d > 2`.
- 0.8.0: synthesis, `generate(fit, n)`.
- 1.0.0: documentation and API polish as the reference release.

Tier 2 (fused kernels for common likelihoods) and Tier 5 (vector and matrix
DSL) become optional, scheduled only when a real user case pulls them.
