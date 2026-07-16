# gpumetropolis 0.4.2

- `gpu_metropolis(method = "de", de_sync = TRUE)` adds the synchronous
  per-generation Differential Evolution path (path B). The population advances
  one generation at a time behind a barrier with a double buffer, so every
  chain's proposal reads the whole previous generation, the canonical DE-MC
  mixing of ter Braak (2006); the difference pair excludes the proposing
  chain. This complements the default batched-snapshot path A, whose pool
  refreshes every `de_every` iterations: path B refreshes it every generation,
  which helps on curved or strongly correlated targets where a stale pool
  loses efficiency. The synchronous path runs on the CPU backend in this
  release; its warmup is trim-only and `proposal_sd` is the fixed jitter base.

# gpumetropolis 0.4.1

- A plotting layer for the joint posterior and the Bayesian decisions.
  `gpum_pairs(fit, crlb = ...)` draws the marginal posteriors with their
  highest density intervals on the diagonal and the bivariate posteriors with
  their credible-region contours off the diagonal, overlaying the Cramer-Rao
  ellipse when a `gpum_crlb` is supplied, so the convergence region of each
  parameter pair reads against the information-bound reference.
  `gpum_region(fit, params)` returns the highest posterior density region of a
  pair as contour polygons that compose onto any plot with `lines()`, including
  a scatter of the original data. `gpum_surface(fit, params)` shows the
  bivariate posterior density as level curves in two dimensions and as a
  three-dimensional surface with the credible-region contours projected on the
  floor.
- Explanatory figures for the hypothesis tools: `plot()` methods for the
  objects of `gpum_hypothesis()` (the posterior with the hypothesis interval
  shaded and its probability), `gpum_rope()` (the posterior with the ROPE band,
  the highest density interval bar and the decision, in the style of Kruschke),
  `gpum_crlb()` (posterior spread against the Cramer-Rao bound) and
  `gpum_bayes_factor()` (the two log evidences with the factor on the Jeffreys
  scale).
- `gpum_ppc()` and `gpum_density_compare()` add the observed-against-generated
  check that applied work needs. `gpum_ppc(fit, generate)` draws a posterior
  predictive sample, with the user supplying the family's one-line simulator
  since the package does not yet generate from an arbitrary likelihood;
  `gpum_density_compare(observed, generated)` overlays the observed density and
  one or more generated densities on a single plot, so a fit can be checked
  against the data and competing models read off against each other.
- Three case-study vignettes on real data exercise the whole package end to
  end. `case_old_faithful` fits a correlated regression and a bimodal mixture
  to the Old Faithful geyser, covering Differential Evolution, parallel
  tempering, the Cramer-Rao reference both where it applies and where it
  declines, the decision verbs and the model-comparison verbs.
  `case_mtcars` is a model-comparison study on fuel economy, weighing a second
  predictor by WAIC, LOO and the Bayes factor. `case_portpirie` fits the Gumbel
  law to annual maximum sea levels, derives the hundred-year return level with
  its credible interval, and compares the extreme-value law against a Normal.

# gpumetropolis 0.4.0

- `gpu_metropolis(method = "de")` adds Differential Evolution MCMC. The
  proposal for each chain is the scaled difference of two other chains,
  `x_c + gamma * (x_a - x_b) + epsilon`, so the proposal aligns with the
  correlation of the target through the ensemble geometry, with no
  explicit covariance and no hand tuning of `proposal_sd` (ter Braak
  2006). The default scale is `gamma = 2.38 / sqrt(2 * n_params)`, the
  value optimal for an approximately Gaussian target; with probability
  0.1 each iteration the scale collapses to 1 for an occasional
  mode-crossing jump. A small per-dimension jitter `de_noise` (default
  `1e-3` of the running per-chain scale) keeps the chain irreducible
  when the ensemble collapses onto a subspace. The path needs at least
  four chains and maps onto the chain axis the package already
  parallelises.
- The implementation is host-orchestrated (path A): the difference pool
  is the population frozen at the start of each batch, refreshed every
  `de_every` iterations (default `max(n_chains, 10)`), and the kernel
  draws the difference pairs internally from that snapshot. Because the
  snapshot is fixed within a batch the chains stay independent during
  the batch, so the block-per-chain kernel is unchanged in structure;
  the proposal increment is symmetric, so the acceptance ratio is the
  density ratio alone. A per-generation synchronous variant ships in 0.4.2 under
  `de_sync = TRUE`.
- `gpum_diagnose(fit)` recognises a DE fit: the per-parameter summary
  covers every chain (there is no cold chain to collapse to), an extra
  row of panels shows the per-chain acceptance and the ensemble spread
  per dimension over the batches, and a hint fires when the population
  spread collapses in any dimension ("Raise de_noise or n_chains").
- `gpum_crlb(fit, data)` adds an optional Cramer-Rao reference. It forms
  the observed Fisher information by a central-difference Hessian of the
  same compiled log-likelihood the sampler used, inverts it to the
  lower bound on the covariance of an unbiased estimator, and reports it
  beside the empirical posterior spread. Under the regularity of the
  Bernstein-von Mises theorem the posterior covariance approaches the
  inverse Fisher information, so a match is a check that the sampler
  recovered the information-bound geometry. The bound is a frequentist,
  asymptotic, prior-free object, and the function refuses to report a
  number rather than mislead when its assumptions visibly fail: a model
  with no data term, a largest R-hat above a threshold (multimodal or
  unconverged), or an observed information that is not positive definite
  (a non-regular geometry or a boundary). `gpum_diagnose(fit, crlb = ...)`
  overlays the asymptotic-normal reference on the density panels.
- A formal Bayesian decision and comparison layer is added. From the draws
  alone: `gpum_hypothesis(fit, parameter, lower, upper)` gives the posterior
  probability of an interval or one-sided hypothesis, and
  `gpum_rope(fit, parameter, rope)` applies the region-of-practical-equivalence
  rule against the highest density interval `hdi()` (Kruschke 2018). For
  predictive model comparison without a marginal likelihood:
  `gpum_waic(fit, data)` (Watanabe 2010) and `gpum_loo(fit, data)`, the latter
  delegating to the `loo` package for Pareto-smoothed importance sampling
  (Vehtari, Gelman and Gabry 2017), both backed by a new pointwise
  log-likelihood path. For the weight of evidence:
  `gpum_evidence(model, data)` estimates the log marginal likelihood by
  thermodynamic integration along power posteriors from the prior to the
  posterior (Gelman and Meng 1998; Friel and Pettitt 2008), and
  `gpum_bayes_factor(model1, model0, data)` forms the Bayes factor from two
  evidences. The thermodynamic integration reuses the existing sampler with no
  kernel change, by raising the compiled likelihood to the rung power in the
  bytecode. The evidence requires a proper prior and carries the
  prior-sensitivity caveat of the Jeffreys-Lindley effect, stated in the
  output. The posterior predictive check and Bayesian p-value are deferred
  until the package can generate replicated data from an arbitrary likelihood.

# gpumetropolis 0.3.0

- `gpu_metropolis(method = "pt")` adds parallel tempering. Each chain
  runs at its own temperature on the same target; between batches a
  swap step proposes exchanges of states between adjacent temperatures,
  with the textbook acceptance ratio
  `exp((log pi(x_{c+1}) - log pi(x_c)) * (1/T_c - 1/T_{c+1}))`. The cold
  chain (`T = 1`) provides the posterior samples; the hot chains feed it
  through swaps. Default ladder is geometric, `T = 1` to `t_max = 10`
  spaced as `beta^{(c-1)}`. Default `swap_every = max(n_chains, 10)`.
  Adaptation continues to work per chain inside PT: each chain's
  proposal scale adapts to its own tempered geometry during warmup.
- Kernel changes for PT: per-chain `temperatures` buffer threaded
  through the CPU-native and CubeCL kernels, with the acceptance ratio
  divided by `T_c`; the kernel also returns the raw log-posterior at
  each chain's final state so the host swap step can compare densities
  across chains without recomputing.
- `gpum_diagnose(fit)` recognises a PT fit: the convergence summary
  collapses to the cold chain, an extra row of panels shows swap
  acceptance per pair over batches with the 0.234 asymptotic target as
  a reference, and a hint fires when any adjacent pair averages below
  10% swap acceptance ("Consider a denser temperature ladder").

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
