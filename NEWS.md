# gpumetropolis 0.2.0

- **`gpu_metropolis(backend = "auto")` is now the default.** The selector
  picks `"cuda"` when its feature is compiled in, then `"vulkan"`, and
  falls back to `"cpu"` with a one-shot per-session notice pointing at
  the source-install recipe.
- **Adaptive warmup, on by default.** When `adapt = TRUE` and `warmup
  > 0`, the warmup runs in batches; between batches, Welford updates the
  per-chain running variance per dimension while Robbins-Monro updates
  a per-chain scalar toward the asymptotic optimum acceptance (0.234 in
  `d >= 2`, 0.44 in `d = 1`; Roberts-Gelman-Gilks 1997). Adaptation
  stops cleanly at the warmup boundary, so the sampling phase is
  stationary. `proposal_sd` becomes a seed for the warmup rather than a
  knob to sintonise; `adapt = FALSE` keeps the trim-only warmup of
  0.1.x. `proposal_sd` also accepts an `n_chains` by `n_params` matrix
  for the explicit per-chain initialisation.
- **`gpum_diagnose(fit)`: one-call diagnostic.** Prints a per-parameter
  table (mean, sd, 2.5%, 50%, 97.5% quantiles, split R-hat, ESS, MCSE)
  and a convergence verdict from the canonical thresholds (R-hat < 1.01
  and ESS >= 400). When `plot = TRUE`, opens a multi-panel plot per
  parameter (trace, pooled density, running mean per chain, pooled
  ACF), plus, when the fit is adaptive, an extra row showing the
  acceptance rate per chain by warmup batch with the asymptotic
  optimum as a reference.
- The fit object now carries `fit$adaptation` when the warmup was
  adaptive, with `final_proposal_sd`, `final_scales`, `n_batches`,
  `batch_sizes` and `accept_history`.

# gpumetropolis 0.1.3

- On attach, the package now notifies the user when a newer version is
  available on the R-universe channel of the maintainer. The check is
  active only in interactive sessions, silent on any network or parse
  failure, and opt-out via the `GPUMETROPOLIS_NO_VERSION_CHECK`
  environment variable. The hint includes the source-install command, so
  the auto-detected GPU backend kicks in on the upgrade.

# gpumetropolis 0.1.2

- `gpu_metropolis()` now discards a warmup portion before returning. The
  new `warmup` argument defaults to `floor(n_iter / 2)`, following the
  convention of Stan and nimble, so `fit$draws` is post-warmup by default
  and is suitable for direct plotting and posterior summaries. Set
  `warmup = 0` to keep every iteration, useful for trace plots that show
  the burn-in trajectory. The trim is plain; an adaptive warmup that also
  tunes the proposal during the burn-in is the next release.
- The `gpum_fit` object now carries `n_iter` (kept), `n_iter_total` (raw)
  and `warmup` so the raw and discarded counts are recoverable.

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
