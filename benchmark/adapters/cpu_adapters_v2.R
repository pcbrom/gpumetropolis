# Spec-driven backend adapters for the M2 to M4 factorial.
#
# These generalise benchmark/adapters/cpu_adapters.R, which is frozen for the
# M1 run, to models of any dimension. Every adapter has the same contract:
#   adapter(spec, data, n_iter, n_chains, seed)
#     -> list(draws, time_sec, meta)
# where `draws` is an array of post-warmup draws with dimensions
# (n_keep, n_chains, spec$dim), `n_keep = n_iter - n_iter %/% 2`, `time_sec`
# is the wall-clock of the sampling call only, and `meta` carries backend and
# version information. The per-model parts of each competitor, the nimble
# model code and the Stan file, are carried by the spec.

# Wall-clock of an expression in seconds, microsecond resolution on Linux.
bench_elapsed <- function(expr) {
  t0 <- Sys.time()
  force(expr)
  as.numeric(difftime(Sys.time(), t0, units = "secs"))
}

# A wide uniform box around the overdispersed starts, used as a flat prior by
# the BayesianTools adapter; the posterior mass sits well inside it.
bench_box <- function(spec, data, n_chains) {
  starts <- spec$init(data, max(n_chains, 8L))
  lo <- apply(starts, 2L, min) - 3
  hi <- apply(starts, 2L, max) + 3
  list(lower = lo, upper = hi)
}

# gpumetropolis through the generic API. The model is declared in the DSL and
# the same declaration drives every backend; only `backend` changes.
make_gpumetropolis_adapter <- function(be) {
  force(be)
  function(spec, data, n_iter, n_chains, seed) {
    n_warmup <- n_iter %/% 2L
    n_keep <- n_iter - n_warmup
    model <- gpumetropolis::gpum_model(spec$gpum_loglik, params = spec$params,
                                       data = spec$data_names)
    init <- spec$init(data, n_chains)
    data_list <- stats::setNames(list(data), spec$data_names)
    fit <- NULL
    elapsed <- bench_elapsed(
      fit <- gpumetropolis::gpu_metropolis(
        model, data = data_list, init = init,
        proposal_sd = spec$proposal_sd(data),
        n_iter = n_iter, warmup = n_warmup, seed = seed,
        backend = be
      )
    )
    draws <- fit$draws
    dim(draws) <- c(n_keep, n_chains, spec$dim)
    list(
      draws = draws, time_sec = elapsed,
      meta = list(backend = paste0("gpumetropolis-", be),
                  version = as.character(utils::packageVersion("gpumetropolis")))
    )
  }
}

# gpumetropolis in parallel-tempering mode. The cold chain (T = 1) is the
# only one that samples the target; the hot chains are auxiliary and feed
# the cold one through swaps. The adapter returns the cold chain
# replicated across `n_chains` slots so the standard harness diagnostics
# (R-hat, ESS) treat the chain count consistently with the other
# backends. A factor-aware harness that distinguishes the cold chain
# from the hot ones is a future refinement.
make_gpumetropolis_pt_adapter <- function(be) {
  force(be)
  function(spec, data, n_iter, n_chains, seed) {
    n_warmup <- n_iter %/% 2L
    n_keep <- n_iter - n_warmup
    model <- gpumetropolis::gpum_model(spec$gpum_loglik, params = spec$params,
                                       data = spec$data_names)
    init <- spec$init(data, n_chains)
    data_list <- stats::setNames(list(data), spec$data_names)
    fit <- NULL
    elapsed <- bench_elapsed(
      fit <- gpumetropolis::gpu_metropolis(
        model, data = data_list, init = init,
        proposal_sd = spec$proposal_sd(data),
        n_iter = n_iter, warmup = n_warmup,
        method = "pt", seed = seed, backend = be
      )
    )
    cold <- fit$draws[, 1L, , drop = FALSE]
    draws <- array(rep(cold, n_chains),
                   dim = c(dim(cold)[1L], n_chains, spec$dim))
    list(
      draws = draws, time_sec = elapsed,
      meta = list(backend = paste0("gpumetropolis-", be, "-pt"),
                  version = as.character(utils::packageVersion("gpumetropolis")),
                  swap_history_mean =
                    if (!is.null(fit$adaptation$swap_history)) {
                      mean(fit$adaptation$swap_history, na.rm = TRUE)
                    } else {
                      NA_real_
                    })
    )
  }
}

# mcmc::metrop. One chain per call; the log-density is an R closure, so this
# backend pays R-callback overhead. Handles a parameter vector of any length.
adapter_mcmc <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  obj <- function(theta) spec$log_post(theta, data)
  inits <- spec$init(data, n_chains)
  scale <- spec$proposal_sd(data)
  draws <- array(0, c(n_keep, n_chains, spec$dim))
  elapsed <- bench_elapsed({
    for (ch in seq_len(n_chains)) {
      set.seed(seed + ch)
      out <- mcmc::metrop(obj, initial = inits[ch, ], nbatch = n_iter,
                          scale = scale)
      draws[, ch, ] <- out$batch[seq.int(n_warmup + 1L, n_iter), , drop = FALSE]
    }
  })
  list(draws = draws, time_sec = elapsed,
       meta = list(backend = "mcmc",
                   version = as.character(utils::packageVersion("mcmc"))))
}

# MCMCpack::MCMCmetrop1R. One chain per call; the package runs a preliminary
# optim for the proposal covariance, kept inside the timed region.
adapter_mcmcpack <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  fun <- function(theta) spec$log_post(theta, data)
  inits <- spec$init(data, n_chains)
  draws <- array(0, c(n_keep, n_chains, spec$dim))
  elapsed <- bench_elapsed({
    for (ch in seq_len(n_chains)) {
      invisible(utils::capture.output(
        chain <- MCMCpack::MCMCmetrop1R(
          fun, theta.init = inits[ch, ], burnin = n_warmup, mcmc = n_keep,
          thin = 1L, tune = 1, verbose = 0, logfun = TRUE,
          seed = as.integer(seed + ch)
        )
      ))
      draws[, ch, ] <- as.matrix(chain)
    }
  })
  list(draws = draws, time_sec = elapsed,
       meta = list(backend = "MCMCpack",
                   version = as.character(utils::packageVersion("MCMCpack"))))
}

# nimble. The model is built and compiled to C++ outside the timed region,
# since protocol section 6.2 excludes compilation from the ESS/s metric. The
# model code is carried by the spec.
adapter_nimble <- function(spec, data, n_iter, n_chains, seed) {
  suppressMessages(requireNamespace("nimble", quietly = TRUE))
  suppressMessages(library(nimble))
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  # spec$nimble_code is the quoted model block; splice it into nimbleCode so
  # its substitute() call captures the block rather than the symbol.
  code <- do.call(nimble::nimbleCode, list(spec$nimble_code))
  inits <- spec$nimble_inits(data, n_chains)
  samples <- NULL
  suppressMessages(invisible(utils::capture.output({
    model <- nimble::nimbleModel(
      code, constants = spec$nimble_constants(data),
      data = spec$nimble_data(data), inits = inits[[1]]
    )
    cmodel <- nimble::compileNimble(model)
    conf <- nimble::configureMCMC(model, monitors = spec$nimble_monitors)
    mcmc <- nimble::buildMCMC(conf)
    cmcmc <- nimble::compileNimble(mcmc, project = model)
  })))
  elapsed <- bench_elapsed(suppressMessages(invisible(utils::capture.output(
    samples <- nimble::runMCMC(
      cmcmc, niter = n_iter, nburnin = n_warmup, nchains = n_chains,
      inits = inits, setSeed = seed + seq_len(n_chains), progressBar = FALSE
    )
  ))))
  # Monitored columns, in the parameter order of spec$params.
  cols <- if (spec$dim == 1L) {
    spec$nimble_monitors
  } else {
    paste0(spec$nimble_monitors, "[", seq_len(spec$dim), "]")
  }
  draws <- array(0, c(n_keep, n_chains, spec$dim))
  for (ch in seq_len(n_chains)) {
    chain <- if (n_chains == 1L) samples else samples[[ch]]
    draws[, ch, ] <- as.matrix(chain[, cols, drop = FALSE])
  }
  list(draws = draws, time_sec = elapsed,
       meta = list(backend = "nimble",
                   version = as.character(utils::packageVersion("nimble"))))
}

# BayesianTools, basic Metropolis sampler. One chain per call; a flat prior is
# approximated by a uniform prior on a wide box.
adapter_bayesiantools <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  ll <- function(par) spec$log_post(par, data)
  box <- bench_box(spec, data, n_chains)
  setup <- BayesianTools::createBayesianSetup(
    likelihood = ll, lower = box$lower, upper = box$upper,
    names = spec$params
  )
  draws <- array(0, c(n_keep, n_chains, spec$dim))
  elapsed <- bench_elapsed({
    for (ch in seq_len(n_chains)) {
      set.seed(seed + ch)
      out <- BayesianTools::runMCMC(
        setup, sampler = "Metropolis",
        settings = list(iterations = n_iter, message = FALSE,
                        consoleUpdates = 0)
      )
      s <- BayesianTools::getSample(out, start = n_warmup + 1L)
      s <- as.matrix(s)
      draws[, ch, ] <- s[seq_len(n_keep), , drop = FALSE]
    }
  })
  list(draws = draws, time_sec = elapsed,
       meta = list(backend = "BayesianTools",
                   version = as.character(utils::packageVersion("BayesianTools"))))
}

# Stan through cmdstanr. The model is compiled once and cached; compilation is
# outside the timed region. Stan uses NUTS; the ESS/s metric is
# algorithm-neutral.
adapter_stan <- function(spec, data, n_iter, n_chains, seed) {
  n_warmup <- n_iter %/% 2L
  n_keep <- n_iter - n_warmup
  mod <- cmdstanr::cmdstan_model(spec$stan_file)
  # nimble_inits returns per-chain lists keyed by the parameter node name,
  # which matches spec$stan_pars for every model here (mu, mu, theta).
  inits <- spec$nimble_inits(data, n_chains)
  fit <- NULL
  elapsed <- bench_elapsed(
    fit <- mod$sample(
      data = spec$stan_data(data), chains = n_chains, parallel_chains = 1L,
      iter_warmup = n_warmup, iter_sampling = n_keep, init = inits,
      seed = seed, refresh = 0L, show_messages = FALSE,
      show_exceptions = FALSE
    )
  )
  arr <- fit$draws(spec$stan_pars)
  draws <- array(as.numeric(arr), dim = c(n_keep, n_chains, spec$dim))
  ndiv <- tryCatch(sum(fit$diagnostic_summary()$num_divergent),
                   error = function(e) NA_real_)
  list(draws = draws, time_sec = elapsed,
       meta = list(backend = "Stan-cmdstanr",
                   version = as.character(utils::packageVersion("cmdstanr")),
                   divergences = ndiv))
}

cpu_adapters <- list(
  "gpumetropolis-cpu" = make_gpumetropolis_adapter("cpu"),
  "gpumetropolis-cuda" = make_gpumetropolis_adapter("cuda"),
  "gpumetropolis-vulkan" = make_gpumetropolis_adapter("vulkan"),
  "gpumetropolis-cpu-pt" = make_gpumetropolis_pt_adapter("cpu"),
  "gpumetropolis-cuda-pt" = make_gpumetropolis_pt_adapter("cuda"),
  "gpumetropolis-vulkan-pt" = make_gpumetropolis_pt_adapter("vulkan"),
  "mcmc" = adapter_mcmc,
  "MCMCpack" = adapter_mcmcpack,
  "nimble" = adapter_nimble,
  "BayesianTools" = adapter_bayesiantools,
  "Stan-cmdstanr" = adapter_stan
)
