#' Batched random-walk Metropolis sampler for a Gaussian mean
#'
#' Samples the posterior of the mean `mu` of observations assumed to follow
#' `Normal(mu, sigma^2)` with known `sigma` and a flat prior on `mu`. Many
#' independent chains are advanced together: at every iteration the candidate
#' state of each chain is evaluated by a single batched call to the log-density
#' kernel.
#'
#' This is the CPU reference sampler. Its log-density kernel runs on the CPU.
#' Later package versions dispatch the same kernel to a GPU and are validated
#' for distributional equivalence against this function with [ks_equivalence()]
#' and [rhat()]. The sequential dependence within a chain (`x_{t+1}` depends on
#' `x_t`) is intrinsic to Markov chain Monte Carlo and is not parallelised; the
#' parallel axes are the chains and the sum over the data.
#'
#' @param data Numeric vector of observations. Must be finite and non-empty.
#' @param sigma Positive finite scalar, the known standard deviation of the
#'   observations.
#' @param n_iter Number of iterations recorded per chain. Default 2000.
#' @param n_chains Number of independent chains. Used only when `init` is
#'   `NULL`. Default 4.
#' @param init Optional numeric vector of starting values, one per chain. When
#'   supplied, its length sets the number of chains. When `NULL`, chains start
#'   evenly spread over `mean(data) +/- 2 * sigma`, which overdisperses them so
#'   that [rhat()] is informative while staying bounded for many chains.
#' @param proposal_sd Positive finite scalar, the standard deviation of the
#'   Gaussian random-walk proposal. When `NULL`, defaults to
#'   `2.4 * sigma / sqrt(length(data))`, scaled to the width of the posterior.
#' @param seed Single integer seed. Each chain runs an independent PCG64 stream
#'   derived from `(seed, chain_index)`, so the result is reproducible and does
#'   not depend on chain scheduling.
#'
#' @return An object of class `gpumetropolis_fit`: a list with `draws` (an
#'   `n_iter` by `n_chains` numeric matrix), `accept_rate` (per-chain
#'   acceptance rate) and the run metadata.
#'
#' @examples
#' set.seed(1)
#' x <- rnorm(500, mean = 3, sd = 2)
#' fit <- metropolis_gaussian_mean(x, sigma = 2, n_iter = 1000)
#' rhat(fit)
#'
#' @seealso [rhat()], [ess()], [ks_equivalence()], [gaussian_mean_posterior()]
#' @export
metropolis_gaussian_mean <- function(data, sigma, n_iter = 2000L,
                                     n_chains = 4L, init = NULL,
                                     proposal_sd = NULL, seed = 1L) {
  if (!is.numeric(data) || length(data) < 1L || anyNA(data) ||
        any(!is.finite(data))) {
    stop("`data` must be a non-empty numeric vector of finite values.",
         call. = FALSE)
  }
  if (!is.numeric(sigma) || length(sigma) != 1L || !is.finite(sigma) ||
        sigma <= 0) {
    stop("`sigma` must be a single positive finite number.", call. = FALSE)
  }
  n_iter <- as.integer(n_iter)
  if (length(n_iter) != 1L || is.na(n_iter) || n_iter < 1L) {
    stop("`n_iter` must be a single positive integer.", call. = FALSE)
  }

  if (is.null(init)) {
    n_chains <- as.integer(n_chains)
    if (length(n_chains) != 1L || is.na(n_chains) || n_chains < 1L) {
      stop("`n_chains` must be a single positive integer.", call. = FALSE)
    }
    init <- if (n_chains == 1L) {
      mean(data)
    } else {
      mean(data) + sigma * seq(-2, 2, length.out = n_chains)
    }
  } else {
    if (!is.numeric(init) || length(init) < 1L || anyNA(init) ||
          any(!is.finite(init))) {
      stop("`init` must be a non-empty numeric vector of finite values.",
           call. = FALSE)
    }
    n_chains <- length(init)
  }

  if (is.null(proposal_sd)) {
    proposal_sd <- 2.4 * sigma / sqrt(length(data))
  }
  if (!is.numeric(proposal_sd) || length(proposal_sd) != 1L ||
        !is.finite(proposal_sd) || proposal_sd <= 0) {
    stop("`proposal_sd` must be a single positive finite number.",
         call. = FALSE)
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
    stop("`seed` must be a single finite number.", call. = FALSE)
  }

  res <- rust_metropolis_gaussian_mean(
    data = as.numeric(data),
    sigma = as.numeric(sigma),
    n_iter = n_iter,
    init = as.numeric(init),
    proposal_sd = as.numeric(proposal_sd),
    seed = as.numeric(seed)
  )

  draws <- res$draws
  colnames(draws) <- paste0("chain", seq_len(n_chains))

  structure(
    list(
      draws = draws,
      accept_rate = res$accept_rate,
      n_iter = n_iter,
      n_chains = n_chains,
      model = "gaussian_mean",
      n_data = length(data),
      sigma = as.numeric(sigma),
      proposal_sd = as.numeric(proposal_sd),
      seed = as.numeric(seed)
    ),
    class = "gpumetropolis_fit"
  )
}

#' @export
print.gpumetropolis_fit <- function(x, ...) {
  cat("<gpumetropolis_fit>\n")
  cat(sprintf("  model       : %s (n = %d, sigma = %g)\n",
              x$model, x$n_data, x$sigma))
  cat(sprintf("  chains      : %d\n", x$n_chains))
  cat(sprintf("  iterations  : %d per chain\n", x$n_iter))
  cat(sprintf("  proposal_sd : %g\n", x$proposal_sd))
  cat(sprintf("  accept_rate : %.3f to %.3f\n",
              min(x$accept_rate), max(x$accept_rate)))
  half <- x$n_iter %/% 2L
  post <- x$draws[seq.int(half + 1L, x$n_iter), , drop = FALSE]
  cat(sprintf("  posterior mu: %.4f (sd %.4f), discarding first %d as warmup\n",
              mean(post), stats::sd(as.vector(post)), half))
  invisible(x)
}
