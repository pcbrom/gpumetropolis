# The dependent-data layer. The engine evaluates a per-row conditional
# log-density and sums the rows, which is exactly the prequential
# factorisation p(y_{1:n}) = p(y_{1:p}) prod_t p(y_t | y_{t-p:t-1}): a
# Markov dependence of finite order enters through lagged columns with no
# change to the sampler. The asymptotics that license the inference change
# name but not shape: the score of the conditional factorisation is a
# martingale difference sequence, so the martingale CLT (Hall and Heyde
# 1980) replaces Lindeberg-Feller, Billingsley (1961) gives the MLE
# asymptotics for ergodic Markov processes, and Borwanker, Kallianpur and
# Prakasa Rao (1971) give the Bernstein-von Mises theorem that backs the
# posterior credible intervals.

#' Declare a Markov (time-series) model by conditional factorisation
#'
#' Builds a [gpum_model()] for dependent data with a Markov dependence of
#' order `p`. The user writes the conditional log-density of one
#' observation given its `p` predecessors, referring to the current value
#' by the series name and to the lagged values as `<name>_lag1` through
#' `<name>_lagp`; `gpum_ts_model()` assembles the lagged design so the
#' engine's row sum computes the conditional log-likelihood
#' `sum_{t = p + 1}^{n} log f(y_t | y_{t-1}, ..., y_{t-p})`.
#'
#' The likelihood conditions on the first `p` observations, the standard
#' conditional-likelihood treatment (Billingsley 1961; Hamilton 1994,
#' section 5.2); for `n` moderately large the information in the first `p`
#' points is negligible. A stationary initial density, when available in
#' closed form, can be added through `prior`, which is evaluated once per
#' parameter vector.
#'
#' Inference downstream is licensed by the dependent-data asymptotics: the
#' conditional score is a martingale difference sequence, so the observed
#' information of the conditional factorisation is the right curvature
#' (martingale CLT, Hall and Heyde 1980) and the posterior satisfies a
#' Bernstein-von Mises theorem for ergodic Markov processes (Borwanker,
#' Kallianpur and Prakasa Rao 1971). Model comparison by [gpum_waic()] or
#' [gpum_loo()] assumes exchangeable pointwise terms and does not transfer
#' to dependent rows; use [gpum_lfo()] instead.
#'
#' @param loglik One-sided formula, the conditional log-density of the
#'   current observation. Refer to the series by its name in `series` and
#'   to lags as `<name>_lag1`, ..., `<name>_lagp`. Exogenous covariates
#'   from `covariates` enter by their own names (aligned to the current
#'   time index).
#' @param params Character vector of parameter names.
#' @param series Named list with one entry, the observed series
#'   (`list(y = <numeric>)`); the name is the symbol used in `loglik`.
#' @param order Markov order `p >= 1`, the number of lags the conditional
#'   density uses.
#' @param covariates Optional named list of exogenous covariate vectors,
#'   each the same length as the series; row `t` of the design carries the
#'   covariate at time `t`.
#' @param prior Optional one-sided formula for the log-prior (plus, when
#'   wanted, the stationary log-density of the initial state), evaluated
#'   once per parameter vector.
#'
#' @return A list of class `gpum_ts` with the compiled `model` (a
#'   `gpum_model`), the assembled `data` list ready for
#'   [gpu_metropolis()], the `order`, and the effective sample rows
#'   `n_rows = length(y) - order`.
#'
#' @examples
#' set.seed(1)
#' y <- as.numeric(arima.sim(list(ar = 0.6), n = 300))
#' ts_m <- gpum_ts_model(
#'   ~ -0.5 * ((y - mu - phi * (y_lag1 - mu)) / exp(ls))^2 - ls,
#'   params = c("mu", "phi", "ls"), series = list(y = y), order = 1,
#'   prior = ~ -0.5 * (mu / 10)^2 - 0.5 * (phi / 2)^2 - 0.5 * (ls / 2)^2
#' )
#' fit <- gpu_metropolis(ts_m$model, data = ts_m$data, n_iter = 2000,
#'                       n_chains = 4, seed = 1)
#'
#' @seealso [gpum_model()], [gpu_metropolis()], [gpum_lfo()]
#' @export
gpum_ts_model <- function(loglik, params, series, order = 1L,
                          covariates = NULL, prior = NULL) {
  if (!is.list(series) || length(series) != 1L || is.null(names(series)) ||
      !nzchar(names(series))) {
    stop("`series` must be a named list with exactly one entry, e.g. ",
         "`list(y = <numeric>)`.", call. = FALSE)
  }
  order <- as.integer(order)
  if (is.na(order) || order < 1L) {
    stop("`order` must be an integer >= 1.", call. = FALSE)
  }
  nm <- names(series)
  y <- as.numeric(series[[1L]])
  n <- length(y)
  if (n <= order + 1L) {
    stop("the series needs more than `order + 1` observations.",
         call. = FALSE)
  }
  rows <- seq.int(order + 1L, n)
  data <- stats::setNames(list(y[rows]), nm)
  for (k in seq_len(order)) {
    data[[paste0(nm, "_lag", k)]] <- y[rows - k]
  }
  if (!is.null(covariates)) {
    if (!is.list(covariates) || is.null(names(covariates)) ||
        any(!nzchar(names(covariates)))) {
      stop("`covariates` must be a named list of vectors.", call. = FALSE)
    }
    for (cn in names(covariates)) {
      v <- as.numeric(covariates[[cn]])
      if (length(v) != n) {
        stop("covariate '", cn, "' must have the length of the series (",
             n, ").", call. = FALSE)
      }
      data[[cn]] <- v[rows]
    }
  }
  model <- gpum_model(loglik, params = params, data = names(data),
                      prior = prior)
  structure(
    list(model = model, data = data, order = order, series_name = nm,
         n_rows = length(rows), n_series = n),
    class = "gpum_ts"
  )
}

#' @export
print.gpum_ts <- function(x, ...) {
  cat("<gpum_ts>\n")
  cat(sprintf("  series     : %s (n = %d)\n", x$series_name, x$n_series))
  cat(sprintf("  Markov order: %d (conditional rows: %d)\n",
              x$order, x$n_rows))
  cat(sprintf("  parameters : %s\n",
              paste(x$model$params, collapse = ", ")))
  cat("  factorisation: conditional on the first `order` observations\n")
  invisible(x)
}

#' Exact leave-future-out cross-validation for dependent data
#'
#' Estimates the expected log predictive density of one-step-ahead
#' forecasts by leave-future-out cross-validation with an expanding
#' window: for each held-out time `t` in the evaluation range, the model
#' is refit on rows `1:(t - 1)` and scored on the conditional log
#' predictive density of row `t`, integrated over the posterior by
#' averaging the pointwise likelihood across the draws. This is the
#' evaluation scheme that respects temporal ordering; [gpum_waic()] and
#' [gpum_loo()] assume exchangeable pointwise contributions and are not
#' valid for dependent rows (Burkner, Gabry and Vehtari 2020). The refits
#' are exact rather than importance-sampling approximations: the sampler
#' is fast enough that the approximation of Burkner, Gabry and Vehtari
#' (2020) buys nothing here.
#'
#' @param ts_model A `gpum_ts` from [gpum_ts_model()].
#' @param L Minimum training rows before the first evaluation: rows
#'   `1:L` train the first refit and row `L + 1` is the first scored
#'   point. Default keeps the last quarter of the rows for evaluation.
#' @param n_iter,n_chains,seed,... Passed to [gpu_metropolis()] for each
#'   refit.
#'
#' @return A list of class `gpum_lfo` with `elpd_lfo` (the summed
#'   expected log predictive density), `se` (its standard error over the
#'   evaluated points), and `pointwise` (one row per evaluated time).
#'
#' @seealso [gpum_ts_model()], [gpum_waic()], [gpum_loo()]
#' @export
gpum_lfo <- function(ts_model, L = NULL, n_iter = 4000L, n_chains = 4L,
                     seed = 1L, ...) {
  if (!inherits(ts_model, "gpum_ts")) {
    stop("`ts_model` must come from gpum_ts_model().", call. = FALSE)
  }
  n_rows <- ts_model$n_rows
  if (is.null(L)) {
    L <- max(ts_model$order + 10L, n_rows - max(10L, n_rows %/% 4L))
  }
  L <- as.integer(L)
  if (L < 2L || L >= n_rows) {
    stop("`L` must lie in [2, n_rows - 1].", call. = FALSE)
  }
  model <- ts_model$model
  data <- ts_model$data
  eval_rows <- seq.int(L + 1L, n_rows)
  lpd <- numeric(length(eval_rows))
  for (i in seq_along(eval_rows)) {
    t_row <- eval_rows[i]
    train <- lapply(data, function(col) col[seq_len(t_row - 1L)])
    fit <- gpu_metropolis(model, data = train, n_iter = n_iter,
                          n_chains = n_chains, seed = seed, ...)
    test <- lapply(data, function(col) col[t_row])
    # draws-by-1 matrix of the conditional log-density of the held-out row.
    ll <- as.numeric(.gpum_pointwise(fit, test))
    # log E_post[f(y_t | past, theta)] via log-sum-exp over the draws.
    m <- max(ll)
    lpd[i] <- m + log(mean(exp(ll - m)))
  }
  out <- list(
    elpd_lfo = sum(lpd),
    se = stats::sd(lpd) * sqrt(length(lpd)),
    pointwise = data.frame(row = eval_rows, elpd = lpd),
    L = L, n_eval = length(eval_rows)
  )
  class(out) <- "gpum_lfo"
  out
}

#' @export
print.gpum_lfo <- function(x, ...) {
  cat("<gpum_lfo> exact leave-future-out cross-validation\n")
  cat(sprintf("  evaluated points : %d (training window from %d rows)\n",
              x$n_eval, x$L))
  cat(sprintf("  elpd_lfo         : %.2f (se %.2f)\n", x$elpd_lfo, x$se))
  invisible(x)
}
