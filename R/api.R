# Generic user-facing API of gpumetropolis.
#
# The user declares a model with `gpum_model()`, giving the log-likelihood and
# the log-prior as one-sided formulas in a restricted operation set, and runs
# it with `gpu_metropolis()`. The formulas are compiled to bytecode (R/dsl.R)
# and executed by the CubeCL interpreter kernel, so the same model runs on the
# CPU and GPU runtimes from one declaration.

#' Declare a model for the GPU-portable Metropolis sampler
#'
#' Compiles a log-likelihood and an optional log-prior, declared as one-sided
#' formulas, into the bytecode the sampler runs. The log-likelihood is a
#' per-observation expression: the sampler sums it over the data. The formulas
#' may use `+`, `-`, `*`, `/`, `^`, unary `-`, and `exp`, `log`, `sqrt`; any
#' other symbol or function is rejected with a clear error.
#'
#' @param loglik One-sided formula, the per-observation log-likelihood, up to
#'   an additive constant. It may reference the parameter names and the data
#'   column names.
#' @param params Character vector of parameter names.
#' @param data Character vector of data column names. Empty for a model with
#'   no data term.
#' @param prior One-sided formula, the joint log-prior over the parameters, up
#'   to an additive constant. It may reference only the parameter names. `NULL`
#'   is a flat prior.
#'
#' @return An object of class `gpum_model`.
#'
#' @examples
#' # Gaussian mean with known sd = 2 and a flat prior on mu.
#' m <- gpum_model(
#'   loglik = ~ -((y - mu)^2) / 8,
#'   params = "mu",
#'   data = "y"
#' )
#'
#' @seealso [gpu_metropolis()]
#' @export
gpum_model <- function(loglik, params, data = character(0), prior = NULL) {
  if (!inherits(loglik, "formula")) {
    stop("`loglik` must be a one-sided formula such as `~ expr`.",
         call. = FALSE)
  }
  params <- as.character(params)
  data <- as.character(data)
  if (length(params) < 1L) {
    stop("`params` must name at least one parameter.", call. = FALSE)
  }
  if (anyDuplicated(c(params, data))) {
    stop("parameter and data names must be distinct.", call. = FALSE)
  }

  ll <- .gpum_compile(.gpum_rhs(loglik), params, data)
  pr <- if (is.null(prior)) {
    list(code = integer(0), consts = numeric(0), depth = 0L)
  } else {
    if (!inherits(prior, "formula")) {
      stop("`prior` must be a one-sided formula or NULL.", call. = FALSE)
    }
    .gpum_compile(.gpum_rhs(prior), params, character(0))
  }

  structure(
    list(loglik = ll, prior = pr, params = params, data = data,
         n_params = length(params), n_cols = length(data)),
    class = "gpum_model"
  )
}

#' @export
print.gpum_model <- function(x, ...) {
  cat("<gpum_model>\n")
  cat(sprintf("  parameters : %s\n", paste(x$params, collapse = ", ")))
  cat(sprintf("  data       : %s\n",
              if (length(x$data)) paste(x$data, collapse = ", ") else "(none)"))
  cat(sprintf("  loglik     : %d bytecode instructions\n",
              length(x$loglik$code) %/% 2L))
  cat(sprintf("  prior      : %s\n",
              if (length(x$prior$code)) {
                sprintf("%d instructions", length(x$prior$code) %/% 2L)
              } else {
                "flat"
              }))
  invisible(x)
}

#' Run the GPU-portable batched Metropolis sampler
#'
#' Advances many independent random-walk Metropolis chains over a model
#' declared with [gpum_model()]. The log-density kernel runs on the chosen
#' backend; the data is uploaded once and stays resident across the run.
#'
#' @param model A `gpum_model` object.
#' @param data A named list or data frame with one entry per data column named
#'   in the model. Ignored for a model with no data term.
#' @param init Optional numeric matrix of starting values, `n_chains` rows by
#'   one column per parameter. When supplied its row count sets the number of
#'   chains. When `NULL`, chains start from independent standard normal draws.
#' @param proposal_sd Initial scale of the Gaussian random-walk proposal.
#'   Accepts a scalar (recycled to all parameters and chains), a length-
#'   `n_params` vector (broadcast across chains) or an `n_chains` by
#'   `n_params` matrix (one row per chain). When `adapt = TRUE`, this is
#'   only the warmup seed and adaptation refines it per chain.
#' @param n_iter Iterations the sampler runs per chain, counting warmup.
#'   Default 2000.
#' @param n_chains Number of chains. Used only when `init` is `NULL`.
#'   Default 4.
#' @param warmup Warmup iterations to discard before returning, following the
#'   convention of Stan and nimble. Must lie in `[0, n_iter)`. Default
#'   `floor(n_iter / 2)`, so `fit$draws` is post-warmup and is suitable for
#'   direct plotting. Set `warmup = 0` to keep every iteration, useful for
#'   inspecting the burn-in trajectory in a trace plot. The string `"auto"`
#'   (adaptive `"rwm"` path) treats `floor(n_iter / 2)` as a budget and stops
#'   the warmup early once the per-chain acceptance has sat at the asymptotic
#'   target for two consecutive batches, spending the remaining iterations on
#'   sampling instead of discarding them.
#' @param adapt Whether to adapt the per-chain proposal scale during
#'   warmup. When `TRUE` (default) and `warmup > 0`, the warmup runs in
#'   batches; between batches, Welford updates the per-chain running
#'   variance and Robbins-Monro updates the per-chain scalar toward the
#'   asymptotic optimum acceptance (0.234 in `d >= 2`, 0.44 in `d = 1`;
#'   Roberts-Gelman-Gilks 1997, Roberts-Rosenthal 2009). Adaptation stops
#'   cleanly at the end of warmup, so the sampling phase is stationary.
#'   Set `FALSE` to keep the trim-only warmup of 0.1.x.
#' @param method Sampling method: `"rwm"` (default) is the random-walk
#'   Metropolis with optional per-chain adaptation; `"pt"` is parallel
#'   tempering with one chain per temperature on the same target and
#'   adjacent-pair swap proposals between batches, where the cold chain
#'   (`T = 1`) provides the posterior samples; `"de"` is Differential
#'   Evolution MCMC, whose proposal for each chain is the scaled
#'   difference of two other chains plus a small jitter, so the proposal
#'   aligns with the correlation of the target through the ensemble
#'   geometry without an explicit covariance (ter Braak 2006). The `"de"`
#'   path needs at least 4 chains. `"mala"` is the Metropolis-adjusted
#'   Langevin algorithm: the proposal drifts along the gradient of the
#'   log-posterior, obtained by reverse-mode automatic differentiation of
#'   the compiled model (JIT-compiled to native code alongside the
#'   density), and the warmup preconditions both drift and noise with the
#'   pooled posterior covariance. The gradient buys mixing that scales as
#'   `d^{-1/3}` against the random walk's `d^{-1/2}`, so `"mala"` is the
#'   method of choice as the dimension grows; it targets acceptance 0.574
#'   (Roberts and Rosenthal 1998) and runs on `backend = "cpu"` in this
#'   release.
#' @param temperatures Numeric vector of length `n_chains`, the
#'   temperature of each chain in `method = "pt"`. Ignored for the other
#'   methods. When `NULL` (default), a geometric ladder from 1 to 10 is
#'   used. The first entry must be 1 for the cold chain.
#' @param swap_every Integer, the iteration count between swap proposals
#'   in `method = "pt"`. Ignored for the other methods. When `NULL`
#'   (default), uses `max(n_chains, 10)`.
#' @param gamma Scale of the difference vector in `method = "de"`.
#'   Ignored for the other methods. When `NULL` (default), uses
#'   `2.38 / sqrt(2 * n_params)`, the value optimal for an approximately
#'   Gaussian target (ter Braak 2006). With probability 0.1 each
#'   iteration the scale collapses to 1 for an occasional mode-crossing
#'   jump.
#' @param de_noise Non-negative factor for the per-dimension jitter added
#'   to the `method = "de"` proposal, in units of the current per-chain
#'   per-dimension scale. A small positive value keeps the chain
#'   irreducible when the ensemble collapses onto a subspace. Default
#'   `1e-3`. Ignored for the other methods.
#' @param de_every Integer, the iteration count between refreshes of the
#'   frozen population snapshot used for the difference vectors in
#'   `method = "de"`. Ignored for the other methods. When `NULL`
#'   (default), uses `max(n_chains, 10)`.
#' @param de_sync Logical. When `TRUE`, `method = "de"` runs the synchronous
#'   per-generation Differential Evolution (path B): the population advances
#'   one generation at a time behind a barrier with a double buffer, so every
#'   proposal reads the previous generation, the canonical DE-MC mixing of
#'   ter Braak (2006). Better per-iteration mixing on curved or strongly
#'   correlated targets, at the cost of a barrier per generation. Currently
#'   CPU only (`backend = "cpu"`); the warmup is trim-only on this path and
#'   `proposal_sd` is the fixed jitter base. Default `FALSE`, the batched
#'   snapshot path A.
#' @param seed Integer seed. Each chain advances its own counter-based stream
#'   from a triple32 hash; the seed is itself hashed, so runs with consecutive
#'   integer seeds get independent streams.
#' @param backend Compute backend: `"cpu"`, `"cuda"` (NVIDIA-native),
#'   `"vulkan"` (vendor-agnostic, through wgpu), or `"auto"`. `"auto"`
#'   selects `"cuda"` when its feature was compiled into the build, then
#'   `"vulkan"`, and falls back to `"cpu"` with a one-shot per-session
#'   message stating that no GPU backend is available. Default `"auto"`.
#'
#' @return An object of class `gpum_fit`: a list with `draws` (an
#'   `n_iter - warmup` by `n_chains` by `n_params` array of post-warmup
#'   samples), `accept_rate`, `n_iter` (kept count), `n_iter_total` (raw
#'   count), `warmup` and the rest of the run metadata.
#'
#' @examples
#' set.seed(1)
#' y <- rnorm(2000, mean = 3, sd = 2)
#' m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
#' fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.05,
#'                       n_iter = 1000, n_chains = 4)
#' rhat(fit$draws[, , 1])
#'
#' @seealso [gpum_model()], [rhat()], [ess()]
#' @export
gpu_metropolis <- function(model, data = NULL, init = NULL, proposal_sd = 0.1,
                           n_iter = 2000L, n_chains = 4L, warmup = NULL,
                           adapt = TRUE, method = c("rwm", "pt", "de", "mala"),
                           temperatures = NULL, swap_every = NULL,
                           gamma = NULL, de_noise = 1e-3, de_every = NULL,
                           de_sync = FALSE,
                           seed = 1L,
                           backend = c("auto", "cpu", "cuda", "vulkan")) {
  # Conjugate fast path: a gpum_lm() model has its posterior in closed form,
  # so the fit is exact independent sampling, no chain, no warmup, no
  # proposal. Every sampler argument other than n_iter, n_chains and seed is
  # meaningless there and is ignored.
  if (inherits(model, "gpum_conjugate")) {
    return(.gpum_exact_fit(model, n_iter = n_iter, n_chains = n_chains,
                           seed = seed))
  }
  if (!inherits(model, "gpum_model")) {
    stop("`model` must be a gpum_model from gpum_model() or a ",
         "gpum_conjugate from gpum_lm().", call. = FALSE)
  }
  backend <- match.arg(backend)
  method <- match.arg(method)

  # The native CPU backend parallelises over chains with a Rayon thread pool,
  # which by default claims every core. Under `R CMD check` the CRAN check
  # farm allows at most two cores; cap the pool there. Rayon reads
  # `RAYON_NUM_THREADS` when the pool is first built, so setting it before the
  # first call into the backend takes effect for the session. A user running
  # outside the check keeps the full pool.
  if (nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_")) &&
      !nzchar(Sys.getenv("RAYON_NUM_THREADS"))) {
    Sys.setenv(RAYON_NUM_THREADS = "2")
  }

  if (model$n_cols > 0L) {
    df <- as.data.frame(data, stringsAsFactors = FALSE)
    missing_cols <- setdiff(model$data, names(df))
    if (length(missing_cols)) {
      stop("missing data columns: ", paste(missing_cols, collapse = ", "),
           call. = FALSE)
    }
    mat <- as.matrix(df[, model$data, drop = FALSE])
    n_obs <- nrow(mat)
    data_flat <- as.vector(t(mat))
  } else {
    n_obs <- 0L
    data_flat <- numeric(0)
  }

  np <- model$n_params
  if (is.null(init)) {
    n_chains <- as.integer(n_chains)
    # Derive the starting values deterministically from `seed`, so the whole
    # run is reproducible from `seed` alone, without disturbing the caller's
    # random number stream.
    if (exists(".Random.seed", envir = .GlobalEnv)) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv)
      on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    }
    set.seed(as.integer(seed))
    init_mat <- matrix(stats::rnorm(n_chains * np), nrow = n_chains, ncol = np)
  } else {
    init_mat <- as.matrix(init)
    if (ncol(init_mat) != np) {
      stop("`init` must have one column per parameter (", np, ").",
           call. = FALSE)
    }
    n_chains <- nrow(init_mat)
  }
  n_iter <- as.integer(n_iter)
  warmup_auto <- identical(warmup, "auto")
  if (is.null(warmup) || warmup_auto) {
    warmup <- n_iter %/% 2L
  }
  warmup <- as.integer(warmup)
  if (is.na(warmup) || warmup < 0L || warmup >= n_iter) {
    stop("`warmup` must be a non-negative integer strictly less than ",
         "`n_iter`.", call. = FALSE)
  }
  # `proposal_sd` accepts three shapes:
  #   - a single scalar, recycled to all parameters and chains;
  #   - a length-`n_params` vector, broadcast across the chains;
  #   - a `n_chains` by `n_params` matrix, one row per chain.
  # The kernel reads a flat length-`n_chains * n_params` buffer in
  # chain-major order, that is `(chain1_dim1, chain1_dim2, ..., chain2_dim1, ...)`.
  proposal_sd_mat <- if (is.matrix(proposal_sd)) {
    if (!all(dim(proposal_sd) == c(n_chains, np))) {
      stop("`proposal_sd` matrix must be `n_chains` by `n_params` (",
           n_chains, " by ", np, ").", call. = FALSE)
    }
    as.matrix(proposal_sd)
  } else {
    matrix(rep_len(as.numeric(proposal_sd), np),
           nrow = n_chains, ncol = np, byrow = TRUE)
  }
  storage.mode(proposal_sd_mat) <- "double"
  if (any(!is.finite(proposal_sd_mat)) || any(proposal_sd_mat <= 0)) {
    stop("`proposal_sd` must be positive and finite.", call. = FALSE)
  }
  proposal_sd_flat <- as.numeric(t(proposal_sd_mat))

  # Resolve "auto" once the chain count is known: a GPU does not help a single
  # chain, so few chains run on the CPU and many chains on a GPU backend, if a
  # GPU backend was compiled into this build.
  avail <- rust_available_backends()
  if (identical(backend, "auto")) {
    if ("cuda" %in% avail) {
      backend <- "cuda"
    } else if ("vulkan" %in% avail) {
      backend <- "vulkan"
    } else {
      backend <- "cpu"
      if (!isTRUE(.gpum_env$auto_cpu_warned)) {
        .gpum_env$auto_cpu_warned <- TRUE
        message("gpumetropolis: no GPU backend in this build, using CPU. ",
                "Install from source with `nvcc` or `vulkaninfo` on PATH ",
                "to enable a GPU backend.")
      }
    }
  } else if (!(backend %in% avail)) {
    stop("backend '", backend, "' is not available in this build. ",
         "Available: ", paste(avail, collapse = ", "),
         ". Rebuild the package with the matching Cargo feature.",
         call. = FALSE)
  }

  if (identical(method, "mala") && !identical(backend, "cpu")) {
    stop("`method = \"mala\"` runs on `backend = \"cpu\"` in this release; ",
         "the Langevin gradient path is not in the GPU kernel yet.",
         call. = FALSE)
  }

  rust_call <- function(init_flat, sd_flat, n_iter, seed,
                        temperatures_flat = rep(1.0, n_chains),
                        proposal_mode = 0L, gamma = 0, de_noise = 0,
                        proposal_l = numeric(0)) {
    rust_gpu_metropolis(
      model$loglik$code, model$loglik$consts, np,
      as.numeric(data_flat), model$n_cols, n_obs,
      model$prior$code, model$prior$consts,
      init_flat, sd_flat,
      as.numeric(temperatures_flat),
      as.integer(n_iter), as.numeric(seed), backend,
      as.integer(proposal_mode), as.numeric(gamma), as.numeric(de_noise),
      as.numeric(proposal_l)
    )
  }

  if (identical(method, "pt")) {
    if (is.null(temperatures)) {
      temperatures <- .pt_default_ladder(n_chains)
    } else {
      temperatures <- as.numeric(temperatures)
      if (length(temperatures) != n_chains) {
        stop("`temperatures` must have one value per chain.", call. = FALSE)
      }
      if (any(!is.finite(temperatures)) || any(temperatures <= 0)) {
        stop("`temperatures` must be positive and finite.", call. = FALSE)
      }
    }
    if (is.null(swap_every)) {
      swap_every <- max(as.integer(n_chains), 10L)
    }
    pt <- .pt_orchestrate(
      rust_call = rust_call, np = np, n_chains = n_chains,
      init_mat = init_mat, proposal_sd_mat = proposal_sd_mat,
      temperatures = temperatures, n_iter = n_iter, warmup = warmup,
      swap_every = swap_every, adapt = isTRUE(adapt),
      seed = as.numeric(seed)
    )
    draws <- array(
      as.numeric(pt$draws),
      dim = dim(pt$draws),
      dimnames = list(NULL, NULL, model$params)
    )
    return(structure(
      list(draws = draws, accept_rate = pt$accept_rate, model = model,
           n_iter = dim(pt$draws)[1L],
           n_iter_total = pt$warmup_used + dim(pt$draws)[1L],
           warmup = pt$warmup_used, n_chains = n_chains,
           n_params = np, backend = backend, seed = seed,
           method = "pt", temperatures = temperatures,
           swap_every = swap_every,
           adaptation = list(
             final_proposal_sd = pt$final_proposal_sd,
             final_scales = pt$final_scales,
             n_batches = pt$n_batches,
             batch_sizes = pt$batch_sizes,
             accept_history = pt$accept_history,
             swap_history = pt$swap_history
           )),
      class = "gpum_fit"
    ))
  }

  if (identical(method, "de")) {
    if (n_chains < 4L) {
      stop("`method = \"de\"` needs at least 4 chains; the proposal draws a ",
           "distinct pair of other chains for each difference vector. ",
           "Increase `n_chains`.", call. = FALSE)
    }
    if (is.null(gamma)) {
      # ter Braak (2006): the scale that is optimal for an approximately
      # Gaussian target of dimension `np`.
      gamma <- 2.38 / sqrt(2 * np)
    }
    gamma <- as.numeric(gamma)
    de_noise <- as.numeric(de_noise)
    if (!is.finite(gamma) || gamma <= 0) {
      stop("`gamma` must be positive and finite.", call. = FALSE)
    }
    if (!is.finite(de_noise) || de_noise < 0) {
      stop("`de_noise` must be non-negative and finite.", call. = FALSE)
    }
    if (is.null(de_every)) {
      de_every <- max(as.integer(n_chains), 10L)
    }
    if (isTRUE(de_sync)) {
      # Path B: synchronous per-generation DE, CPU native only. The Rust side
      # loops the generations behind a barrier with a double buffer; the host
      # trims the warmup rows, so this path is trim-only (no Welford).
      if (!identical(backend, "cpu")) {
        stop("`de_sync = TRUE` runs on `backend = \"cpu\"` in this release; ",
             "the synchronous path needs a per-generation barrier that the ",
             "block-per-chain GPU kernel does not expose yet.", call. = FALSE)
      }
      res <- rust_gpu_metropolis_de_sync(
        model$loglik$code, model$loglik$consts, np,
        as.numeric(data_flat), model$n_cols, n_obs,
        model$prior$code, model$prior$consts,
        as.numeric(t(init_mat)), proposal_sd_flat,
        as.integer(n_iter), as.numeric(seed),
        as.numeric(gamma), as.numeric(de_noise)
      )
      draws <- array(
        res$draws,
        dim = c(res$n_iter, res$n_chains, res$n_params),
        dimnames = list(NULL, NULL, model$params)
      )
      kept <- res$n_iter - warmup
      if (warmup > 0L) {
        draws <- draws[seq.int(warmup + 1L, res$n_iter), , , drop = FALSE]
      }
      return(structure(
        list(draws = draws, accept_rate = res$accept_rate, model = model,
             n_iter = kept, n_iter_total = res$n_iter, warmup = warmup,
             n_chains = res$n_chains, n_params = np, backend = backend,
             seed = seed, method = "de", gamma = gamma, de_noise = de_noise,
             de_sync = TRUE),
        class = "gpum_fit"
      ))
    }
    de <- .de_orchestrate(
      rust_call = rust_call, np = np, n_chains = n_chains,
      init_mat = init_mat, proposal_sd_mat = proposal_sd_mat,
      gamma = gamma, de_noise = de_noise, n_iter = n_iter, warmup = warmup,
      de_every = as.integer(de_every), adapt = isTRUE(adapt),
      seed = as.numeric(seed)
    )
    draws <- array(
      as.numeric(de$draws), dim = dim(de$draws),
      dimnames = list(NULL, NULL, model$params)
    )
    return(structure(
      list(draws = draws, accept_rate = de$accept_rate, model = model,
           n_iter = dim(de$draws)[1L],
           n_iter_total = de$warmup_used + dim(de$draws)[1L],
           warmup = de$warmup_used, n_chains = n_chains,
           n_params = np, backend = backend, seed = seed,
           method = "de", gamma = gamma, de_noise = de_noise,
           de_every = de_every,
           adaptation = list(
             final_proposal_sd = de$final_proposal_sd,
             n_batches = de$n_batches,
             batch_sizes = de$batch_sizes,
             accept_history = de$accept_history,
             disp_history = de$disp_history
           )),
      class = "gpum_fit"
    ))
  }

  langevin <- identical(method, "mala")
  if (isTRUE(adapt) && warmup > 0L) {
    am <- .am_orchestrate_warmup(
      rust_call = rust_call, np = np, n_chains = n_chains,
      init_mat = init_mat, proposal_sd = proposal_sd_mat,
      warmup = warmup, seed = as.numeric(seed),
      auto_stop = warmup_auto, langevin = langevin
    )
    sampling_iters <- n_iter - am$warmup_used
    if (sampling_iters < 1L) sampling_iters <- 1L
    # The sampling phase carries the full-covariance proposal the warmup
    # estimated: `state + L z` with the per-chain Cholesky factor frozen at
    # the warmup boundary, so the phase is stationary. Under MALA the same
    # factor is the preconditioner of the Langevin drift and noise.
    use_L <- np > 1L && !is.null(am$final_L)
    res <- rust_call(
      init_flat = as.numeric(t(am$final_init)),
      sd_flat = as.numeric(t(am$final_proposal_sd)),
      n_iter = sampling_iters,
      seed = am$seed_next,
      proposal_mode = if (langevin) 3L else if (use_L) 2L else 0L,
      proposal_l = if (use_L) am$final_L else numeric(0)
    )
    draws <- array(
      res$draws,
      dim = c(res$n_iter, res$n_chains, res$n_params),
      dimnames = list(NULL, NULL, model$params)
    )
    return(structure(
      list(draws = draws, accept_rate = res$accept_rate, model = model,
           n_iter = res$n_iter, n_iter_total = am$warmup_used + res$n_iter,
           warmup = am$warmup_used, n_chains = res$n_chains,
           n_params = res$n_params, backend = backend, seed = seed,
           method = if (langevin) "mala" else "rwm",
           adaptation = list(
             final_proposal_sd = am$final_proposal_sd,
             final_scales = am$final_scales,
             n_batches = am$n_batches,
             batch_sizes = am$batch_sizes,
             accept_history = am$accept_history
           )),
      class = "gpum_fit"
    ))
  }

  res <- rust_call(
    init_flat = as.numeric(t(init_mat)),
    sd_flat = proposal_sd_flat,
    n_iter = n_iter,
    seed = as.numeric(seed),
    proposal_mode = if (langevin) 3L else 0L
  )

  draws <- array(
    res$draws,
    dim = c(res$n_iter, res$n_chains, res$n_params),
    dimnames = list(NULL, NULL, model$params)
  )
  kept <- res$n_iter - warmup
  if (warmup > 0L) {
    draws <- draws[seq.int(warmup + 1L, res$n_iter), , , drop = FALSE]
  }
  structure(
    list(draws = draws, accept_rate = res$accept_rate, model = model,
         n_iter = kept, n_iter_total = res$n_iter, warmup = warmup,
         n_chains = res$n_chains, n_params = res$n_params,
         backend = backend, seed = seed,
         method = if (langevin) "mala" else "rwm"),
    class = "gpum_fit"
  )
}

#' @export
print.gpum_fit <- function(x, ...) {
  cat("<gpum_fit>\n")
  cat(sprintf("  parameters  : %s\n", paste(x$model$params, collapse = ", ")))
  cat(sprintf("  method      : %s\n", x$method %||% "rwm"))
  cat(sprintf("  backend     : %s\n", x$backend))
  cat(sprintf("  chains      : %d\n", x$n_chains))
  mode <- if (!is.null(x$adaptation)) "adaptive" else "trim"
  cat(sprintf("  iterations  : %d per chain (%d raw, %d %s warmup discarded)\n",
              x$n_iter, x$n_iter_total, x$warmup, mode))
  cat(sprintf("  accept_rate : %.3f to %.3f\n",
              min(x$accept_rate), max(x$accept_rate)))
  if (identical(x$method, "pt") && !is.null(x$adaptation$swap_history)) {
    sh <- x$adaptation$swap_history
    sw_mean <- rowMeans(sh, na.rm = TRUE)
    cat(sprintf("  swap accept : pairs (1-%d) mean %.3f to %.3f\n",
                length(sw_mean), min(sw_mean, na.rm = TRUE),
                max(sw_mean, na.rm = TRUE)))
  }
  if (identical(x$method, "de") && !is.null(x$gamma)) {
    cat(sprintf("  de scale    : gamma %.4f, noise factor %.1e\n",
                x$gamma, x$de_noise %||% 0))
  }
  for (j in seq_len(x$n_params)) {
    if (identical(x$method, "pt")) {
      post <- x$draws[, 1L, j]
      label <- sprintf("%s (T=1)", x$model$params[j])
    } else {
      post <- x$draws[, , j]
      label <- x$model$params[j]
    }
    cat(sprintf("  %-11s : posterior mean %.4f (sd %.4f)\n",
                label, mean(post), stats::sd(as.vector(post))))
  }
  invisible(x)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
