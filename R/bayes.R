# Formal Bayesian decision and comparison tools over a fit. Three layers:
# estimation decisions from the draws alone (posterior probability of a
# hypothesis, ROPE and HDI), predictive comparison from the pointwise
# log-likelihood (WAIC and PSIS-LOO), and evidence by thermodynamic
# integration (the marginal likelihood and the Bayes factor). The posterior
# predictive check and Bayesian p-value are deferred until the package can
# generate replicated data from an arbitrary likelihood.

# Pooled posterior draws of one parameter, the cold chain under PT.
.gpum_posterior_vec <- function(fit, parameter) {
  if (!parameter %in% fit$model$params) {
    stop("`parameter` '", parameter, "' is not a model parameter.",
         call. = FALSE)
  }
  is_pt <- identical(fit$method, "pt")
  d <- if (is_pt) fit$draws[, 1L, parameter] else fit$draws[, , parameter]
  as.vector(d)
}

# Pooled posterior draws of every parameter as an n_draws by n_params matrix.
.gpum_posterior_matrix <- function(fit) {
  is_pt <- identical(fit$method, "pt")
  np <- fit$n_params
  pdraws <- if (is_pt) fit$draws[, 1L, , drop = FALSE] else fit$draws
  pch <- if (is_pt) 1L else fit$n_chains
  vapply(seq_len(np), function(p) {
    as.vector(.gpum_param_matrix(pdraws, p, pch))
  }, numeric(dim(pdraws)[1L] * pch))
}

# Flat data buffer and observation count for a model, from a data list.
.gpum_data_flat <- function(model, data) {
  if (model$n_cols == 0L) return(list(flat = numeric(0), n_obs = 0L))
  if (is.null(data)) {
    stop("`data` is required for a model with a data term.", call. = FALSE)
  }
  df <- as.data.frame(data, stringsAsFactors = FALSE)
  mat <- as.matrix(df[, model$data, drop = FALSE])
  list(flat = as.vector(t(mat)), n_obs = nrow(mat))
}

.logsumexp <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}

#' Highest posterior density interval
#'
#' The shortest interval that contains a probability mass `ci` of the
#' samples, found by scanning every window of the sorted draws.
#'
#' @param x Numeric samples.
#' @param ci Probability mass in the interval. Default 0.95.
#' @return A named numeric vector `c(lower, upper)`.
#' @seealso [gpum_rope()]
#' @export
hdi <- function(x, ci = 0.95) {
  x <- sort(as.numeric(x))
  n <- length(x)
  if (n < 2L) stop("`x` needs at least two samples.", call. = FALSE)
  if (ci <= 0 || ci >= 1) stop("`ci` must be in (0, 1).", call. = FALSE)
  m <- max(1L, floor(ci * n))
  if (m >= n) return(c(lower = x[1L], upper = x[n]))
  widths <- x[(m):n] - x[1:(n - m + 1L)]
  i <- which.min(widths)
  c(lower = x[i], upper = x[i + m - 1L])
}

#' Posterior probability of a hypothesis on one parameter
#'
#' Reports the posterior probability that a parameter lies in an interval,
#' below it and above it, directly from the draws. This is an estimation
#' decision: it needs no marginal likelihood and inherits the robustness of
#' the posterior the sampler returned.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param parameter Name of the parameter.
#' @param lower,upper Bounds of the hypothesis interval. Defaults span the
#'   real line, so `lower = 0` gives the one-sided `P(theta > 0 | y)` as
#'   `prob`, the mass in the interval `(0, Inf)`; `prob_below` is then the
#'   complementary `P(theta <= 0 | y)`.
#' @return An object of class `gpum_hypothesis` with `prob` (mass in the
#'   interval), `prob_below`, `prob_above` and the inputs.
#' @seealso [gpum_rope()]
#' @export
gpum_hypothesis <- function(fit, parameter, lower = -Inf, upper = Inf) {
  d <- .gpum_posterior_vec(fit, parameter)
  structure(list(
    parameter = parameter, lower = lower, upper = upper,
    prob = mean(d > lower & d < upper),
    prob_below = mean(d <= lower),
    prob_above = mean(d >= upper),
    n = length(d),
    draws = d
  ), class = "gpum_hypothesis")
}

#' @export
print.gpum_hypothesis <- function(x, ...) {
  cat("<gpum_hypothesis>\n")
  cat(sprintf("  parameter %s, interval (%g, %g)\n",
              x$parameter, x$lower, x$upper))
  cat(sprintf("  P(in interval) = %.4f\n", x$prob))
  if (is.finite(x$lower)) cat(sprintf("  P(<= %g)       = %.4f\n",
                                      x$lower, x$prob_below))
  if (is.finite(x$upper)) cat(sprintf("  P(>= %g)       = %.4f\n",
                                      x$upper, x$prob_above))
  invisible(x)
}

#' ROPE and HDI decision on one parameter
#'
#' Applies the region-of-practical-equivalence rule (Kruschke 2018): the
#' parameter is practically equivalent to the null when its highest density
#' interval lies inside the ROPE, practically different when the interval lies
#' entirely outside it, and undecided otherwise. The decision is from the
#' draws and needs no marginal likelihood.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param parameter Name of the parameter.
#' @param rope Either the half-width of a symmetric ROPE around `null`, or a
#'   length-two vector giving the ROPE interval directly.
#' @param null Centre of a symmetric ROPE. Ignored when `rope` is an interval.
#'   Default 0.
#' @param ci Mass of the highest density interval. Default 0.95.
#' @return An object of class `gpum_rope` with the `hdi`, the `rope` interval,
#'   the share `pct_in_rope` of the posterior inside it, and a `decision`.
#' @seealso [hdi()], [gpum_hypothesis()]
#' @export
gpum_rope <- function(fit, parameter, rope, null = 0, ci = 0.95) {
  d <- .gpum_posterior_vec(fit, parameter)
  rope_int <- if (length(rope) == 1L) null + c(-1, 1) * abs(rope) else {
    sort(as.numeric(rope))
  }
  h <- hdi(d, ci)
  pct_in <- mean(d >= rope_int[1L] & d <= rope_int[2L])
  decision <- if (h[["upper"]] < rope_int[1L] || h[["lower"]] > rope_int[2L]) {
    "different: HDI outside the ROPE"
  } else if (h[["lower"]] >= rope_int[1L] && h[["upper"]] <= rope_int[2L]) {
    "equivalent: HDI within the ROPE"
  } else {
    "undecided: HDI straddles the ROPE"
  }
  structure(list(
    parameter = parameter, hdi = h, rope = rope_int, ci = ci,
    pct_in_rope = pct_in, decision = decision, draws = d
  ), class = "gpum_rope")
}

#' @export
print.gpum_rope <- function(x, ...) {
  cat("<gpum_rope>\n")
  cat(sprintf("  parameter %s\n", x$parameter))
  cat(sprintf("  %.0f%% HDI    [%.4g, %.4g]\n", 100 * x$ci,
              x$hdi[["lower"]], x$hdi[["upper"]]))
  cat(sprintf("  ROPE        [%.4g, %.4g]\n", x$rope[1L], x$rope[2L]))
  cat(sprintf("  in ROPE     %.1f%%\n", 100 * x$pct_in_rope))
  cat(sprintf("  decision    %s\n", x$decision))
  invisible(x)
}

# Pointwise log-likelihood matrix, draws by observations, optionally thinned.
.gpum_pointwise <- function(fit, data, max_draws = 4000L) {
  model <- fit$model
  if (model$n_cols == 0L) {
    stop("WAIC and LOO need a data term; this model has none.", call. = FALSE)
  }
  d <- .gpum_data_flat(model, data)
  pts <- .gpum_posterior_matrix(fit)
  if (nrow(pts) > max_draws) {
    pts <- pts[round(seq(1, nrow(pts), length.out = max_draws)), , drop = FALSE]
  }
  flat <- rust_loglik_pointwise(model$loglik$code, model$loglik$consts,
                                fit$n_params, as.numeric(d$flat),
                                model$n_cols, d$n_obs, as.numeric(t(pts)))
  # rust returns point-major: out[p * n_obs + i]; reshape to obs by draws then
  # transpose to draws by obs, the layout WAIC and loo expect.
  t(matrix(flat, nrow = d$n_obs))
}

#' Widely applicable information criterion (WAIC)
#'
#' Computes WAIC from the pointwise log-likelihood of the fit (Watanabe 2010;
#' Vehtari, Gelman and Gabry 2017). WAIC estimates out-of-sample predictive
#' accuracy without a marginal likelihood, so it is robust where Bayes factors
#' are prior-sensitive; lower WAIC is preferred when comparing models on the
#' same data.
#'
#' The comparison is valid only when the log-likelihood is normalised. The
#' model DSL evaluates the log-density up to an additive constant, which is
#' enough for sampling a fixed model but not for comparing models with
#' different likelihoods, since the dropped constants differ. To compare
#' models with WAIC or LOO, write the full normalised per-observation
#' log-likelihood in the formula, including the normalising constant such as
#' `-0.5 * log(2 * pi * sigma^2)` for a Gaussian.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param data The same data passed to [gpu_metropolis()].
#' @param max_draws Cap on the number of posterior draws used, to bound the
#'   pointwise matrix. Default 4000.
#' @return An object of class `gpum_waic` with `waic`, `elpd_waic` (the
#'   expected log pointwise predictive density), `p_waic` (the effective number
#'   of parameters), the standard error `se` and the `pointwise` terms.
#' @seealso [gpum_loo()], [gpum_bayes_factor()]
#' @export
gpum_waic <- function(fit, data, max_draws = 4000L) {
  ll <- .gpum_pointwise(fit, data, max_draws)
  S <- nrow(ll)
  lppd <- apply(ll, 2L, function(col) .logsumexp(col) - log(S))
  pwaic <- apply(ll, 2L, stats::var)
  elpd_i <- lppd - pwaic
  n <- length(elpd_i)
  structure(list(
    waic = -2 * sum(elpd_i),
    elpd_waic = sum(elpd_i),
    p_waic = sum(pwaic),
    se = sqrt(n * stats::var(elpd_i)),
    n_obs = n, n_draws = S,
    pointwise = elpd_i
  ), class = "gpum_waic")
}

#' @export
print.gpum_waic <- function(x, ...) {
  cat("<gpum_waic>\n")
  cat(sprintf("  WAIC       %.2f (se %.2f)\n", x$waic, 2 * x$se))
  cat(sprintf("  elpd_waic  %.2f\n", x$elpd_waic))
  cat(sprintf("  p_waic     %.2f\n", x$p_waic))
  cat(sprintf("  %d observations, %d draws\n", x$n_obs, x$n_draws))
  invisible(x)
}

#' Pareto-smoothed importance-sampling leave-one-out (PSIS-LOO)
#'
#' Computes PSIS-LOO cross-validation by delegating to the `loo` package, which
#' implements the Pareto-smoothed importance-sampling estimator and its
#' diagnostics (Vehtari, Gelman and Gabry 2017). Like WAIC it estimates
#' out-of-sample predictive accuracy without a marginal likelihood, and the
#' Pareto `k` diagnostic flags observations where the estimate is unreliable.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param data The same data passed to [gpu_metropolis()].
#' @param max_draws Cap on the number of posterior draws used. Default 4000.
#' @return The `loo` object returned by `loo::loo()`.
#' @seealso [gpum_waic()]
#' @export
gpum_loo <- function(fit, data, max_draws = 4000L) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("`gpum_loo()` needs the 'loo' package; install it with ",
         "install.packages(\"loo\").", call. = FALSE)
  }
  ll <- .gpum_pointwise(fit, data, max_draws)
  loo::loo(ll)
}

# Build a power-posterior model whose likelihood is raised to `beta`. The
# per-observation log-likelihood bytecode gains a trailing "multiply by beta"
# (PUSH_CONST beta; MUL), so its summed contribution becomes beta times the
# log-likelihood; the prior is untouched. At beta = 0 the target is the prior,
# at beta = 1 the posterior. This reuses the existing sampler with no kernel
# change.
.gpum_power_model <- function(model, beta) {
  code <- model$loglik$code
  consts <- model$loglik$consts
  idx <- length(consts)            # zero-based index of the appended beta
  model$loglik$code <- as.integer(c(code, 0L, idx, 5L, 0L))
  model$loglik$consts <- as.numeric(c(consts, beta))
  model
}

#' Log marginal likelihood by thermodynamic integration
#'
#' Estimates the log marginal likelihood (the model evidence) by
#' thermodynamic integration along a path of power posteriors
#' `p_beta(theta) proportional to p(y | theta)^beta p(theta)` from the prior
#' (`beta = 0`) to the posterior (`beta = 1`). The log evidence is the integral
#' over `beta` of the expected log-likelihood under `p_beta` (Gelman and Meng
#' 1998; Friel and Pettitt 2008), evaluated on a power-law grid that is dense
#' near the prior, where the integrand is most curved. Each rung is sampled by
#' the package's own sampler, so no kernel change is needed.
#'
#' The evidence requires a proper prior: the marginal likelihood is an integral
#' against the prior, undefined for a flat or absent prior. The function stops
#' when the model has no prior term. The estimate is sensitive to the prior, a
#' property of the marginal likelihood itself, not of the sampler.
#'
#' @param model A `gpum_model` with a proper `prior`.
#' @param data The data for the model.
#' @param n_rungs Number of intervals in the `beta` grid. Default 24.
#' @param power Exponent of the power-law grid `beta = (k / n_rungs)^power`.
#'   Default 5, the value that controls the discretisation bias of trapezoidal
#'   thermodynamic integration.
#' @param n_iter,n_chains,warmup,method,seed,backend Passed to
#'   [gpu_metropolis()] for the per-rung sampling. Defaults are 4000 iterations,
#'   8 chains, `method = "de"`.
#' @return An object of class `gpum_evidence` with `log_evidence`, the `betas`
#'   grid and the `expected_loglik` per rung.
#' @seealso [gpum_bayes_factor()], [gpum_waic()]
#' @export
gpum_evidence <- function(model, data, n_rungs = 24L, power = 5,
                          n_iter = 4000L, n_chains = 8L, warmup = NULL,
                          method = "de", seed = 1L, backend = "cpu") {
  if (!inherits(model, "gpum_model")) {
    stop("`model` must be a gpum_model from gpum_model().", call. = FALSE)
  }
  if (length(model$prior$code) == 0L) {
    stop("Evidence requires a proper prior; this model has none. The marginal ",
         "likelihood is undefined for a flat or absent prior.", call. = FALSE)
  }
  betas <- (seq.int(0L, n_rungs) / n_rungs)^power
  expected <- vapply(betas, function(b) {
    pm <- .gpum_power_model(model, b)
    fit <- gpu_metropolis(pm, data = data, n_iter = n_iter,
                          n_chains = n_chains, warmup = warmup,
                          method = method, seed = seed, backend = backend)
    pts <- .gpum_posterior_matrix(fit)
    d <- .gpum_data_flat(model, data)
    # Mean of the unscaled log-likelihood under the power posterior.
    ll <- rust_loglik_batch(model$loglik$code, model$loglik$consts,
                            model$n_params, as.numeric(d$flat),
                            model$n_cols, d$n_obs, as.numeric(t(pts)))
    mean(ll)
  }, numeric(1))
  # Trapezoidal integral of the expected log-likelihood over the beta grid.
  db <- diff(betas)
  log_evidence <- sum(0.5 * (expected[-1L] + expected[-length(expected)]) * db)
  structure(list(
    log_evidence = log_evidence, betas = betas, expected_loglik = expected,
    n_rungs = n_rungs, power = power
  ), class = "gpum_evidence")
}

#' @export
print.gpum_evidence <- function(x, ...) {
  cat("<gpum_evidence>\n")
  cat(sprintf("  log marginal likelihood %.3f\n", x$log_evidence))
  cat(sprintf("  thermodynamic integration over %d rungs (power %g)\n",
              x$n_rungs, x$power))
  invisible(x)
}

#' Bayes factor between two models by thermodynamic integration
#'
#' Computes the Bayes factor `B_10 = p(y | model1) / p(y | model0)` from the
#' log marginal likelihoods of the two models, each by [gpum_evidence()]. The
#' Bayes factor is the formal Bayesian weight of evidence for one model over
#' another; both models must carry a proper prior.
#'
#' The Bayes factor is sensitive to the priors, sharply so for a point null
#' under a diffuse prior, the Jeffreys-Lindley effect: a more diffuse prior on
#' the larger model penalises it regardless of the data. Report the priors with
#' the factor, and prefer [gpum_waic()] or [gpum_loo()] when the goal is
#' predictive comparison rather than a weight of evidence.
#'
#' @param model1,model0 Two `gpum_model` objects with proper priors, the
#'   alternative and the null.
#' @param data The data, shared by both models.
#' @param ... Passed to [gpum_evidence()] for both models.
#' @return An object of class `gpum_bayes_factor` with `bf10`, `log_bf10`, the
#'   two log evidences and a verbal `interpretation` on the Jeffreys scale.
#' @seealso [gpum_evidence()], [gpum_waic()], [gpum_loo()]
#' @export
gpum_bayes_factor <- function(model1, model0, data, ...) {
  e1 <- gpum_evidence(model1, data, ...)
  e0 <- gpum_evidence(model0, data, ...)
  log_bf <- e1$log_evidence - e0$log_evidence
  bf <- exp(log_bf)
  scale <- function(b) {
    if (b < 1) return(paste0("favours model0 (", scale(1 / b), " reversed)"))
    if (b < 3) "barely worth mention"
    else if (b < 10) "substantial"
    else if (b < 30) "strong"
    else if (b < 100) "very strong"
    else "decisive"
  }
  structure(list(
    bf10 = bf, log_bf10 = log_bf,
    log_evidence1 = e1$log_evidence, log_evidence0 = e0$log_evidence,
    interpretation = scale(bf)
  ), class = "gpum_bayes_factor")
}

#' @export
print.gpum_bayes_factor <- function(x, ...) {
  cat("<gpum_bayes_factor>\n")
  cat(sprintf("  B_10 = %.4g (log %.3f)\n", x$bf10, x$log_bf10))
  cat(sprintf("  log evidence: model1 %.3f, model0 %.3f\n",
              x$log_evidence1, x$log_evidence0))
  cat(sprintf("  evidence for model1 over model0: %s\n", x$interpretation))
  cat("  Note: Bayes factors are prior-sensitive (Jeffreys-Lindley); ",
      "report the priors.\n", sep = "")
  invisible(x)
}
