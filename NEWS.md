# gpumetropolis 0.1.1

- Auto-detect GPU backends at install time. The `configure` step probes the
  build host for the CUDA toolkit (`nvcc`) and the Vulkan tools
  (`vulkaninfo`) and adds the matching Cargo features to the build, so a
  source install on a machine with the toolchains present produces a binary
  that exposes the GPU backends without the user passing any flag. Hosts
  with no GPU toolchain build CPU-only, unchanged. Override with the
  `GPUMETROPOLIS_BACKENDS`, `GPUMETROPOLIS_CUDA` and `GPUMETROPOLIS_VULKAN`
  environment variables.

# gpumetropolis 0.1.0

First release. The package is distributed through R-universe as a focused
generic sampler.

## Sampler

- `gpum_model()` declares a model from a log-likelihood and an optional
  log-prior written as one-sided formulas, in a restricted operation set
  (`+`, `-`, `*`, `/`, `^`, unary `-`, `exp`, `log`, `sqrt`). The formulas are
  compiled to a stack-machine bytecode.
- `gpu_metropolis()` runs a batched random-walk Metropolis sampler over the
  compiled model. One kernel source runs on three backends: `cpu`, `cuda` and
  `vulkan`. `backend = "auto"` selects the CPU for few chains and CUDA for
  many.
- The GPU kernel assigns one block of threads to each chain; the threads of a
  block share the sum over observations through a tree reduction. The CPU
  backend is native Rust, with the log-likelihood JIT-compiled to machine code
  and worked in double precision.
- Each chain advances a counter-based stream from a triple32 hash. The `seed`
  argument is itself hashed before the chain offset is applied, so runs with
  consecutive integer seeds get independent streams rather than streams that
  overlap by a one-counter shift.

## Diagnostics

- `rhat()`, the split potential scale reduction factor.
- `ess()`, the effective sample size by Geyer's initial positive sequence.
- `ks_equivalence()`, a two-sample Kolmogorov-Smirnov check of distributional
  equivalence, thinned to the effective sample size.
- `gaussian_mean_posterior()`, the closed-form posterior used to check
  parameter recovery.
