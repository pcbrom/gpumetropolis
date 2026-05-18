# gpumetropolis roadmap

This file records the direction beyond the first CRAN release. It is a
roadmap, not a commitment: the priority is to ship a focused v0.x to CRAN
first. The items below are tiered by how they relate to the current scope and
to the binding decisions of `~/.claude/plans/gpumetropolis-sprint.md`.

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

This is the first application built on the engine rather than a change to the
engine. It is recorded so the direction is not lost, and it is explicitly
sequenced after the package is a recognised reference as a generic sampler.

The engine is already close to one useful slice: Bayesian inference of the
parameter of an extreme-value copula, the Gumbel family, from pseudo-
observations. The Gumbel copula density is inside the DSL operation set, and
the DSL already accepts several data columns. A bivariate demonstration is a
vignette away and needs no engine change.

Tiered honestly:

- Copula parameter inference, bivariate, extreme-value family, with pseudo-
  observations as data. In reach with the current DSL.
- ABC-MCMC: a Metropolis-Hastings accept step that compares simulated and
  observed summary statistics, for fitting a copula or a generator to a vector
  of dependence measures, Kendall's tau, Spearman's rho, tail-dependence
  coefficients, rather than to a likelihood. The accept step stays inside the
  Metropolis-Hastings family, so it does not reopen decision #1. ABC simulates
  a full data set per step, the regime the many-chains GPU engine serves.
- A Bayesian latent-variable generative model, a Gaussian-copula factor model
  or a latent-Gaussian model: the generative, autoencoder-shaped object whose
  inference engine is MCMC rather than an amortised encoder. It needs the
  vector and matrix DSL support of Tier 5.

Recorded boundary: a neural generative model, a variational autoencoder or a
normalising-flow copula, is a different computational paradigm, gradient
training and automatic differentiation. It is not built on this engine and is
not part of this package's arc. If that route is wanted it is a separate
project with a separate stack.

## Sequencing

The priority order is fixed: be a recognised reference as a generic
vendor-agnostic sampler first, expand scope second. v0.x ships to CRAN as the
focused generic sampler. Tier 1 is the first post-release work. Tier 2
follows. The copula and synthetic-data application direction is taken up only
after the package is established in its current niche. Tiers 3 to 6 are
reviewed when v0.x has real users and a concrete demand.
