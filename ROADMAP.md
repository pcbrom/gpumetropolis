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

## Sequencing

v0.x ships to CRAN as the focused generic sampler. Tier 1 is the first
post-release work. Tier 2 follows. Tiers 3 to 6 are reviewed when v0.x has
real users and a concrete demand.
