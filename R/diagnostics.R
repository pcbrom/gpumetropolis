# Distributional equivalence harness.
#
# These functions are the reference against which later GPU samplers are
# checked. Equivalence for Markov chain Monte Carlo is distributional, never
# bit-exact: two samplers are treated as equivalent when their chains sample
# the same target distribution, judged by the split R-hat statistic, the
# effective sample size and the two-sample Kolmogorov-Smirnov test.

# Return the post-warmup draws of a fit, matrix or vector as a matrix with one
# column per chain. A bare numeric vector is treated as a single chain.
.kept_matrix <- function(x, warmup = NULL) {
  is_vector <- is.numeric(x) && !is.matrix(x) &&
    !inherits(x, "gpumetropolis_fit")
  if (inherits(x, "gpumetropolis_fit")) {
    x <- x$draws
  }
  if (is.numeric(x) && !is.matrix(x)) {
    x <- matrix(x, ncol = 1L)
  }
  if (!is.matrix(x) || !is.numeric(x)) {
    stop("input must be a gpumetropolis_fit, a numeric matrix or a numeric ",
         "vector.", call. = FALSE)
  }
  n <- nrow(x)
  if (is.null(warmup)) {
    warmup <- if (is_vector) 0L else n %/% 2L
  }
  warmup <- as.integer(warmup)
  if (is.na(warmup) || warmup < 0L || warmup >= n) {
    stop("`warmup` must be in [0, n_iter).", call. = FALSE)
  }
  x[seq.int(warmup + 1L, n), , drop = FALSE]
}

# Effective sample size of one chain by Geyer's initial positive sequence.
.ess_chain <- function(v) {
  n <- length(v)
  if (n < 8L || stats::var(v) == 0) {
    return(as.numeric(n))
  }
  lag_max <- min(n - 1L, 2000L)
  rho <- stats::acf(v, lag.max = lag_max, plot = FALSE,
                    demean = TRUE)$acf[-1L, 1L, 1L]
  # Sum autocorrelations in consecutive pairs; keep pairs while the pair sum
  # stays positive (Geyer 1992), which truncates the noisy tail of the ACF.
  m <- length(rho) %/% 2L
  if (m < 1L) {
    return(as.numeric(n))
  }
  pair <- rho[2L * seq_len(m) - 1L] + rho[2L * seq_len(m)]
  neg <- which(pair <= 0)
  last <- if (length(neg)) neg[1L] - 1L else m
  if (last < 1L) {
    return(as.numeric(n))
  }
  tau <- 1 + 2 * sum(pair[seq_len(last)])
  if (tau < 1) tau <- 1
  min(n / tau, n)
}

# Evenly spaced subsample of `v` down to `k` elements.
.subsample <- function(v, k) {
  k <- max(1L, min(as.integer(k), length(v)))
  if (k >= length(v)) {
    return(v)
  }
  v[round(seq(1, length(v), length.out = k))]
}

#' Effective sample size
#'
#' Estimates the effective sample size of a set of chains: the number of
#' independent draws that would carry the same information as the
#' autocorrelated Markov chain Monte Carlo output. Each chain is reduced by
#' Geyer's initial positive sequence estimator and the per-chain values are
#' summed.
#'
#' @param x A `gpumetropolis_fit` object, an `n_iter` by `n_chains` numeric
#'   matrix, or a numeric vector of draws.
#' @param warmup Number of initial iterations discarded before the estimate.
#'   When `NULL`, half of the iterations are discarded for matrix and fit
#'   inputs, and none for vector inputs.
#'
#' @return A single numeric value, the summed effective sample size.
#'
#' @references Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
#'   Statistical Science 7(4), 473-483. \doi{10.1214/ss/1177011137}.
#'
#' @examples
#' set.seed(1)
#' x <- rnorm(800, mean = 0, sd = 1)
#' fit <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 2000)
#' ess(fit)
#'
#' @export
ess <- function(x, warmup = NULL) {
  kept <- .kept_matrix(x, warmup)
  sum(apply(kept, 2L, .ess_chain))
}

#' Split R-hat convergence diagnostic
#'
#' Computes the split potential scale reduction factor for a set of chains.
#' Each chain is split in half and the halves are treated as separate chains,
#' which makes the statistic sensitive to non-stationarity within a chain. A
#' value near 1 is consistent with the chains having reached the same
#' distribution; values above roughly 1.01 to 1.1 indicate that they have not.
#'
#' @param x A `gpumetropolis_fit` object or an `n_iter` by `n_chains` numeric
#'   matrix of draws.
#' @param warmup Number of initial iterations discarded before the split. When
#'   `NULL`, half of the iterations are discarded.
#'
#' @return A single numeric value, the split R-hat statistic.
#'
#' @references Gelman, A. and Rubin, D. B. (1992). Inference from iterative
#'   simulation using multiple sequences. Statistical Science 7(4), 457-472.
#'   \doi{10.1214/ss/1177011136}.
#'
#' @examples
#' set.seed(1)
#' x <- rnorm(500, mean = 0, sd = 1)
#' fit <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 1000)
#' rhat(fit)
#'
#' @export
rhat <- function(x, warmup = NULL) {
  draws <- if (inherits(x, "gpumetropolis_fit")) x$draws else x
  if (!is.matrix(draws) || !is.numeric(draws)) {
    stop("`x` must be a gpumetropolis_fit or a numeric matrix.", call. = FALSE)
  }
  n <- nrow(draws)
  if (is.null(warmup)) warmup <- n %/% 2L
  warmup <- as.integer(warmup)
  if (is.na(warmup) || warmup < 0L || warmup >= n) {
    stop("`warmup` must be in [0, n_iter).", call. = FALSE)
  }
  kept <- draws[seq.int(warmup + 1L, n), , drop = FALSE]
  n_kept <- nrow(kept)
  half <- n_kept %/% 2L
  if (half < 2L) {
    stop("need at least 4 post-warmup iterations per chain for split R-hat.",
         call. = FALSE)
  }
  # Split every chain into a lower and an upper half of equal length `half`.
  lower <- kept[seq_len(half), , drop = FALSE]
  upper <- kept[seq.int(n_kept - half + 1L, n_kept), , drop = FALSE]
  splits <- cbind(lower, upper)

  sub_means <- colMeans(splits)
  sub_vars <- apply(splits, 2L, stats::var)
  between <- half * stats::var(sub_means)
  within <- mean(sub_vars)
  if (within <= 0) {
    return(NaN)
  }
  var_hat <- ((half - 1) * within + between) / half
  sqrt(var_hat / within)
}

#' Distributional equivalence by the two-sample Kolmogorov-Smirnov test
#'
#' Pools the post-warmup draws of two samplers and applies the two-sample
#' Kolmogorov-Smirnov test. It is the equivalence gate for the package: a CPU
#' and a GPU sampler are treated as equivalent when the test does not reject
#' the hypothesis that their draws come from the same distribution.
#'
#' The Kolmogorov-Smirnov test assumes independent draws, while Markov chain
#' Monte Carlo output is autocorrelated; feeding it raw draws makes the test
#' reject far too often. By default the pooled draws are therefore thinned down
#' to the effective sample size returned by [ess()] before the test is applied.
#'
#' Not rejecting is evidence consistent with equivalence, not a proof that the
#' two distributions are identical. Report the statistic and the p-value, and
#' read `equivalent` as "no detected difference at level `alpha`".
#'
#' @param x,y Each a `gpumetropolis_fit` object, an `n_iter` by `n_chains`
#'   numeric matrix, or a numeric vector of draws.
#' @param warmup Number of initial iterations discarded from `x` and `y` before
#'   pooling. When `NULL`, half of the iterations are discarded for matrix and
#'   fit inputs, and none for vector inputs.
#' @param alpha Significance level of the test. Default 0.05.
#' @param thin Controls thinning before the test. `TRUE` (default) thins each
#'   pooled sample to its effective sample size. A positive integer keeps every
#'   `thin`-th draw. `FALSE` applies no thinning.
#'
#' @return A list with the test `statistic`, the `p_value`, the chosen `alpha`,
#'   a logical `equivalent` (`TRUE` when `p_value > alpha`) and the post-thinning
#'   pooled sample sizes `n_x` and `n_y`.
#'
#' @examples
#' set.seed(1)
#' x <- rnorm(800, mean = 1, sd = 1)
#' a <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 2000, seed = 1)
#' b <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 2000, seed = 2)
#' ks_equivalence(a, b)
#'
#' @export
ks_equivalence <- function(x, y, warmup = NULL, alpha = 0.05, thin = TRUE) {
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
        alpha <= 0 || alpha >= 1) {
    stop("`alpha` must be a single number in (0, 1).", call. = FALSE)
  }
  mx <- .kept_matrix(x, warmup)
  my <- .kept_matrix(y, warmup)
  px <- as.vector(mx)
  py <- as.vector(my)

  if (isTRUE(thin)) {
    px <- .subsample(px, floor(sum(apply(mx, 2L, .ess_chain))))
    py <- .subsample(py, floor(sum(apply(my, 2L, .ess_chain))))
  } else if (is.numeric(thin) && length(thin) == 1L && thin > 1) {
    px <- px[seq.int(1L, length(px), by = as.integer(thin))]
    py <- py[seq.int(1L, length(py), by = as.integer(thin))]
  }

  test <- suppressWarnings(stats::ks.test(px, py))
  list(
    statistic = unname(test$statistic),
    p_value = test$p.value,
    alpha = alpha,
    equivalent = test$p.value > alpha,
    n_x = length(px),
    n_y = length(py)
  )
}

#' Closed-form posterior of the Gaussian mean
#'
#' Returns the analytic posterior of the mean `mu` for observations following
#' `Normal(mu, sigma^2)` with known `sigma` and a flat prior on `mu`. The
#' posterior is `Normal(mean(data), sigma^2 / length(data))`. It provides the
#' ground truth used to check that [metropolis_gaussian_mean()] recovers known
#' parameters.
#'
#' @param data Numeric vector of observations.
#' @param sigma Positive finite scalar, the known standard deviation.
#'
#' @return A list with the posterior `mean` and standard deviation `sd`.
#'
#' @examples
#' set.seed(1)
#' gaussian_mean_posterior(rnorm(500, mean = 3, sd = 2), sigma = 2)
#'
#' @export
gaussian_mean_posterior <- function(data, sigma) {
  if (!is.numeric(data) || length(data) < 1L || anyNA(data) ||
        any(!is.finite(data))) {
    stop("`data` must be a non-empty numeric vector of finite values.",
         call. = FALSE)
  }
  if (!is.numeric(sigma) || length(sigma) != 1L || !is.finite(sigma) ||
        sigma <= 0) {
    stop("`sigma` must be a single positive finite number.", call. = FALSE)
  }
  list(mean = mean(data), sd = sigma / sqrt(length(data)))
}
