<!-- [![CRAN_Status_Badge](https://www.r-pkg.org/badges/last-release/gpumetropolis)](https://cran.r-project.org/package=gpumetropolis) -->
<!-- [![Downloads from the RStudio CRAN mirror](https://cranlogs.r-pkg.org/badges/grand-total/gpumetropolis)](https://cran.r-project.org/package=gpumetropolis) -->
<!-- the CRAN badges above are enabled once the package is published -->

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# gpumetropolis

<!-- badges: start -->
<!-- badges: end -->

A generic Metropolis-Hastings sampler for Markov chain Monte Carlo. The user
declares a model by writing its log-likelihood and log-prior as ordinary R
formulas; the package compiles them to a portable kernel that runs on the CPU
and the GPU from one source. The sampler advances many independent chains in
one batched pass. It occupies a niche that is currently empty on CRAN: no CRAN
package offers generic MCMC with a vendor-agnostic GPU-portable kernel.

The model expression is compiled to a stack-machine bytecode that a single
CubeCL kernel interprets, so any model in the supported operation set runs on
the CPU and GPU with no runtime code generation. CubeCL compiles that one
kernel for CUDA, ROCm, Vulkan and CPU back ends.

The package is under active development. See `## Project status` below.

## Installation

The package is not on CRAN yet. The development version requires a Rust
toolchain (`cargo`, `rustc >= 1.85`); see <https://rustup.rs>. The CUDA backend
additionally needs the CUDA toolkit. Install from GitHub with:

``` r
# install.packages("remotes")
remotes::install_github("pcbrom/gpumetropolis")
```

## Load package

``` r
library(gpumetropolis)
```

## Quick start

A model is declared by writing its per-observation log-likelihood as a
one-sided formula in the parameter and data names. The example below is the
Gaussian mean with known standard deviation 2 and a flat prior; its posterior
is available in closed form, which makes it a clean check that the sampler
recovers known parameters.

``` r
set.seed(1)
y <- rnorm(20000, mean = 3.4, sd = 2)

model <- gpum_model(
  loglik = ~ -((y - mu)^2) / 8,   # sigma = 2, so 0.5 / sigma^2 = 1 / 8
  params = "mu",
  data = "y"
)

fit <- gpu_metropolis(model, data = list(y = y), proposal_sd = 0.05,
                      n_iter = 3000, n_chains = 8, backend = "cpu")
fit

# the same declaration runs on the GPU
fit_gpu <- gpu_metropolis(model, data = list(y = y), proposal_sd = 0.05,
                          n_iter = 3000, n_chains = 8, backend = "cuda")
```

The supported operations in a formula are `+`, `-`, `*`, `/`, `^`, unary `-`,
and `exp`, `log`, `sqrt`. A symbol that is not a declared parameter or data
column, or a function outside this set, is rejected at compile time with a
clear error.

## Convergence diagnostics

The package ships a distributional equivalence harness. Equivalence for MCMC is
distributional, never bit-exact, because the algorithm is stochastic.

``` r
draws <- fit$draws[, , "mu"]   # iterations by chains
rhat(draws)                    # split potential scale reduction factor
ess(draws)                     # effective sample size, Geyer estimator

# distributional equivalence between the CPU and GPU runs
ks_equivalence(fit$draws[, , "mu"], fit_gpu$draws[, , "mu"])
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
  the distributional equivalence harness.
- Phase 1, complete: the generic model DSL, the CubeCL bytecode interpreter
  kernel, and the `gpum_model()` / `gpu_metropolis()` API. A model declared by
  formula runs on the CPU and CUDA backends from one source and recovers known
  posteriors; `R CMD check --as-cran` passes with no error.
- Phase 2, next: multi-vendor portability, the AMD GPU through the Vulkan back
  end, and the registered benchmark experiment.
- Phase 3: CRAN submission, including vendoring the Rust dependency tree.

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
Measured results are added to this section once the registered run completes
on the generic package.

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
