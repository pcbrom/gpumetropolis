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

# gpumetropolis through the generic API. The model is declared in the DSL and
# the sampler runs on the chosen backend (cpu, cuda or vulkan). The same
# declaration drives every backend; only `backend` changes.
make_gpumetropolis_adapter <- function(be) {
  force(be)
  function(spec, data, n_iter, n_chains, seed) {
    n_warmup <- n_iter %/% 2L
    model <- gpumetropolis::gpum_model(spec$gpum_loglik, params = "mu",
                                       data = "y")
    init <- matrix(bench_init(data, spec$sigma, n_chains), ncol = 1L)
    psd <- 2.4 * spec$sigma / sqrt(length(data))
    fit <- NULL
    elapsed <- bench_elapsed(
      fit <- gpumetropolis::gpu_metropolis(
        model, data = list(y = data), init = init, proposal_sd = psd,
        n_iter = n_iter, seed = seed, backend = be
      )
    )
    draws <- fit$draws[seq.int(n_warmup + 1L, n_iter), , 1L, drop = FALSE]
    dim(draws) <- c(n_iter - n_warmup, n_chains)
    list(
      draws = draws,
      time_sec = elapsed,
      meta = list(
        backend = paste0("gpumetropolis-", be),
        version = as.character(utils::packageVersion("gpumetropolis"))
      )
    )
  }
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

# nimble. The model is built and compiled to C++ outside the timed region,
# since protocol section 6.2 excludes compilation from the ESS/s metric. The
# log-density is compiled, so this backend does not pay R-callback overhead.
adapter_nimble <- function(spec, data, n_iter, n_chains, seed) {
  # nimble relies on its package being attached; the namespace-qualified form
  # alone does not set up its model-building environment.
  suppressMessages(requireNamespace("nimble", quietly = TRUE))
  suppressMessages(library(nimble))
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  code <- nimble::nimbleCode({
    for (i in 1:N) {
      y[i] ~ dnorm(mu, sd = sigma)
    }
    mu ~ dflat()
  })
  inits <- lapply(bench_init(data, spec$sigma, n_chains),
                  function(m) list(mu = m))
  suppressMessages(invisible(utils::capture.output({
    model <- nimble::nimbleModel(
      code, constants = list(N = length(data), sigma = spec$sigma),
      data = list(y = data), inits = inits[[1]]
    )
    cmodel <- nimble::compileNimble(model)
    conf <- nimble::configureMCMC(model, monitors = "mu")
    mcmc <- nimble::buildMCMC(conf)
    cmcmc <- nimble::compileNimble(mcmc, project = model)
  })))
  samples <- NULL
  elapsed <- bench_elapsed(suppressMessages(invisible(utils::capture.output(
    samples <- nimble::runMCMC(
      cmcmc, niter = n_iter, nburnin = n_warmup, nchains = n_chains,
      inits = inits, setSeed = seed + seq_len(n_chains), progressBar = FALSE
    )
  ))))
  draws <- matrix(0, n_keep, n_chains)
  if (n_chains == 1L) {
    draws[, 1L] <- as.numeric(samples[, "mu"])
  } else {
    for (ch in seq_len(n_chains)) {
      draws[, ch] <- as.numeric(samples[[ch]][, "mu"])
    }
  }
  list(
    draws = draws,
    time_sec = elapsed,
    meta = list(backend = "nimble",
                version = as.character(utils::packageVersion("nimble")))
  )
}

# BayesianTools, basic Metropolis sampler. One chain per call; chains are
# looped. A flat prior is approximated by a uniform prior on a wide box.
adapter_bayesiantools <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  ll <- function(par) spec$log_post(par, data)
  centre <- mean(data)
  setup <- BayesianTools::createBayesianSetup(
    likelihood = ll,
    lower = centre - 10 * spec$sigma,
    upper = centre + 10 * spec$sigma,
    names = "mu"
  )
  draws <- matrix(0, n_keep, n_chains)
  elapsed <- bench_elapsed({
    for (ch in seq_len(n_chains)) {
      set.seed(seed + ch)
      out <- BayesianTools::runMCMC(
        setup, sampler = "Metropolis",
        settings = list(iterations = n_iter, message = FALSE,
                        consoleUpdates = 0)
      )
      s <- as.numeric(BayesianTools::getSample(out, start = n_warmup + 1L))
      draws[, ch] <- s[seq_len(n_keep)]
    }
  })
  list(
    draws = draws,
    time_sec = elapsed,
    meta = list(backend = "BayesianTools",
                version = as.character(utils::packageVersion("BayesianTools")))
  )
}

# Stan through cmdstanr. The model is compiled once and cached by cmdstanr;
# compilation is outside the timed region. Stan uses NUTS, a different
# algorithm; the ESS/s metric is algorithm-neutral. Chains run sequentially
# (parallel_chains = 1) so the timing stays single-threaded.
adapter_stan <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  mod <- cmdstanr::cmdstan_model("benchmark/models/m1.stan")
  inits <- lapply(bench_init(data, spec$sigma, n_chains),
                  function(m) list(mu = m))
  standata <- list(N = length(data), y = data, sigma = spec$sigma)
  fit <- NULL
  elapsed <- bench_elapsed(
    fit <- mod$sample(
      data = standata, chains = n_chains, parallel_chains = 1L,
      iter_warmup = n_warmup, iter_sampling = n_keep, init = inits,
      seed = seed, refresh = 0L, show_messages = FALSE,
      show_exceptions = FALSE
    )
  )
  draws <- matrix(as.numeric(fit$draws("mu")), nrow = n_keep, ncol = n_chains)
  ndiv <- tryCatch(sum(fit$diagnostic_summary()$num_divergent),
                   error = function(e) NA_real_)
  list(
    draws = draws,
    time_sec = elapsed,
    meta = list(backend = "Stan-cmdstanr",
                version = as.character(utils::packageVersion("cmdstanr")),
                divergences = ndiv)
  )
}

cpu_adapters <- list(
  "gpumetropolis-cpu" = make_gpumetropolis_adapter("cpu"),
  "gpumetropolis-cuda" = make_gpumetropolis_adapter("cuda"),
  "gpumetropolis-vulkan" = make_gpumetropolis_adapter("vulkan"),
  "mcmc" = adapter_mcmc,
  "MCMCpack" = adapter_mcmcpack,
  "nimble" = adapter_nimble,
  "BayesianTools" = adapter_bayesiantools,
  "Stan-cmdstanr" = adapter_stan
)
