<!-- [![CRAN_Status_Badge](https://www.r-pkg.org/badges/last-release/gpumetropolis)](https://cran.r-project.org/package=gpumetropolis) -->
<!-- [![Downloads from the RStudio CRAN mirror](https://cranlogs.r-pkg.org/badges/grand-total/gpumetropolis)](https://cran.r-project.org/package=gpumetropolis) -->
<!-- the CRAN badges above are enabled once the package is published -->

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# gpumetropolis

<!-- badges: start -->
<!-- badges: end -->

A Metropolis-Hastings sampler for Markov chain Monte Carlo whose log-density
evaluation kernel is written to be portable across GPU vendors and CPU back
ends from a single source. The sampler advances many independent chains in one
batched pass, evaluating the candidate states of all chains with a single
kernel call. It occupies a niche that is currently empty on CRAN: no CRAN
package offers generic MCMC with a vendor-agnostic GPU-portable kernel.

The package is under active development. This version ships the CPU reference
sampler and the distributional equivalence harness against which the GPU
kernel is validated. See `## Project status` below.

## Installation

The package is not on CRAN yet. The development version requires a Rust
toolchain (`cargo`, `rustc >= 1.65`); see <https://rustup.rs>. Install from
GitHub with:

``` r
# install.packages("remotes")
remotes::install_github("pcbrom/gpumetropolis")
```

## Load package

``` r
library(gpumetropolis)
```

## Quick start

The reference model samples the posterior of the mean of normally distributed
observations with a known standard deviation and a flat prior. The posterior
is available in closed form, which makes it a clean target for checking that
the sampler recovers known parameters.

``` r
set.seed(1)
x <- rnorm(50000, mean = 4.2, sd = 1.7)

fit <- metropolis_gaussian_mean(x, sigma = 1.7, n_iter = 6000, n_chains = 8)
fit

# compare against the analytic posterior
gaussian_mean_posterior(x, sigma = 1.7)
```

## Convergence diagnostics

The package ships the equivalence harness that later GPU versions are checked
against. Equivalence for MCMC is distributional, never bit-exact, because the
algorithm is stochastic.

``` r
rhat(fit)            # split potential scale reduction factor
ess(fit)             # effective sample size, Geyer estimator

# distributional equivalence between two runs
a <- metropolis_gaussian_mean(x, sigma = 1.7, n_iter = 6000, seed = 1)
b <- metropolis_gaussian_mean(x, sigma = 1.7, n_iter = 6000, seed = 2)
ks_equivalence(a, b)
```

`ks_equivalence` thins the pooled draws down to the effective sample size
before the Kolmogorov-Smirnov test, because that test assumes independent
draws while MCMC output is autocorrelated.

## When the GPU helps, and when it does not

A GPU does not accelerate every MCMC. The sequential dependence inside a chain
cannot be parallelised. The parallelism comes from two axes: many independent
chains, and the data-parallel evaluation of the log-density. A GPU pays off
when the log-density is expensive to evaluate, that is over a large data set,
or when thousands of chains are run. For a small model with few chains the
CPU-GPU transfer overhead dominates and the GPU is slower than the CPU. The
package documentation states this regime plainly rather than promising
unconditional speedups.

## Project status

The development follows a phased plan.

- Phase 0, complete: CPU reference sampler in Rust, batched over chains, plus
  the distributional equivalence harness. `R CMD check --as-cran` passes with
  no error and no package warning.
- Phase 1, next: the log-density kernel rewritten in CubeCL, dispatched to the
  GPU, validated for distributional equivalence against the CPU sampler.
- Phase 2: multi-vendor portability across NVIDIA and AMD.
- Phase 3: CRAN submission.

## Benchmark

The package carries a pre-registered experiment that characterises, in a
refutable way, the regime in which `gpumetropolis` beats, ties or loses to the
established R MCMC packages (`MCMCpack`, `mcmc`, `nimble`, `BayesianTools`,
`greta`, Stan via `cmdstanr`). The design, the six hypotheses with their
support and refutation conditions, and the decision rules are frozen in
[`EXPERIMENT_PROTOCOL.md`](https://github.com/pcbrom/gpumetropolis/blob/main/EXPERIMENT_PROTOCOL.md),
committed before any result existed.

The primary metric is effective sample size per second, computed uniformly
with `coda` so the estimator is not a confounder. A correctness gate precedes
any speed comparison: the speed of an incorrect sampler is not reported.

Execution is in two stages. The CPU stage is in progress; the GPU stage
follows Phase 1. Measured results are added to this section once the
registered run completes. Pilot runs so far have validated the harness and the
correctness gate; they are pipeline checks, not the registered experiment, and
their numbers are not reported here as results.

The machine and software environment of the benchmark host is recorded in
[`benchmark/ENVIRONMENT.md`](https://github.com/pcbrom/gpumetropolis/blob/main/benchmark/ENVIRONMENT.md),
regenerated by `benchmark/capture_env.sh`, so the run can be reproduced or
audited.

## Issues

Please report issues at <https://github.com/pcbrom/gpumetropolis/issues>.

## Citation

``` r
citation("gpumetropolis")
```
