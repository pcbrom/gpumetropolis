# CPU backend adapters for stage A of the benchmark.
#
# Every adapter has the same contract:
#   adapter(spec, data, n_iter, n_chains, seed)
#     -> list(draws, time_sec, meta)
# where `draws` is an n_keep by n_chains matrix of post-warmup draws,
# `n_keep = n_iter - n_iter %/% 2`, `time_sec` is the wall-clock of the
# sampling call only, and `meta` carries backend and version information.
#
# Warmup is half the iterations for every sampler, per EXPERIMENT_PROTOCOL.md
# section 7. ESS is computed downstream and uniformly, so the adapters return
# raw draws and never an efficiency figure.

# Wall-clock of an expression in seconds, using a high-resolution clock.
# system.time has roughly millisecond resolution, which rounds the fastest
# cells to zero; Sys.time resolves microseconds on Linux.
bench_elapsed <- function(expr) {
  t0 <- Sys.time()
  force(expr)
  as.numeric(difftime(Sys.time(), t0, units = "secs"))
}

# Overdispersed starting values shared by every adapter, so a speed gap is not
# an artefact of different starts. Matches the default of metropolis_gaussian_mean.
bench_init <- function(data, sigma, n_chains) {
  if (n_chains == 1L) {
    mean(data)
  } else {
    mean(data) + sigma * seq(-2, 2, length.out = n_chains)
  }
}

# gpumetropolis CPU reference sampler. The log-density runs in compiled Rust
# with no per-iteration R callback.
adapter_gpumetropolis_cpu <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  fit <- NULL
  elapsed <- bench_elapsed(
    fit <- gpumetropolis::metropolis_gaussian_mean(
      data = data, sigma = spec$sigma, n_iter = n_iter,
      n_chains = n_chains, seed = seed
    )
  )
  draws <- fit$draws[seq.int(n_warmup + 1L, n_iter), , drop = FALSE]
  list(
    draws = draws,
    time_sec = elapsed,
    meta = list(
      backend = "gpumetropolis-CPU",
      version = as.character(utils::packageVersion("gpumetropolis")),
      accept_rate = mean(fit$accept_rate)
    )
  )
}

# mcmc::metrop. One chain per call; chains are looped. The log-density is an R
# closure called every iteration, so this backend pays R-callback overhead.
adapter_mcmc <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  obj <- function(mu) spec$log_post(mu, data)
  inits <- bench_init(data, spec$sigma, n_chains)
  scale <- 2.4 * spec$sigma / sqrt(length(data))
  draws <- matrix(0, n_keep, n_chains)
  elapsed <- bench_elapsed({
    for (ch in seq_len(n_chains)) {
      set.seed(seed + ch)
      out <- mcmc::metrop(obj, initial = inits[ch], nbatch = n_iter,
                          scale = scale)
      draws[, ch] <- out$batch[seq.int(n_warmup + 1L, n_iter), 1L]
    }
  })
  list(
    draws = draws,
    time_sec = elapsed,
    meta = list(backend = "mcmc",
                version = as.character(utils::packageVersion("mcmc")))
  )
}

# MCMCpack::MCMCmetrop1R. One chain per call; chains are looped. The package
# runs a preliminary optim for the proposal covariance; that cost is part of
# how the package works and stays inside the timed region.
adapter_mcmcpack <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  fun <- function(mu) spec$log_post(mu, data)
  inits <- bench_init(data, spec$sigma, n_chains)
  draws <- matrix(0, n_keep, n_chains)
  elapsed <- bench_elapsed({
    for (ch in seq_len(n_chains)) {
      invisible(utils::capture.output(
        chain <- MCMCpack::MCMCmetrop1R(
          fun, theta.init = inits[ch], burnin = n_warmup, mcmc = n_keep,
          thin = 1L, tune = 1, verbose = 0, logfun = TRUE,
          seed = as.integer(seed + ch)
        )
      ))
      draws[, ch] <- as.numeric(chain)
    }
  })
  list(
    draws = draws,
    time_sec = elapsed,
    meta = list(backend = "MCMCpack",
                version = as.character(utils::packageVersion("MCMCpack")))
  )
}

cpu_adapters <- list(
  "gpumetropolis-CPU" = adapter_gpumetropolis_cpu,
  "mcmc" = adapter_mcmc,
  "MCMCpack" = adapter_mcmcpack
)
