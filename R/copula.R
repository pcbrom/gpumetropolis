# The bivariate copula layer. Sklar's theorem factorises a joint
# distribution into its marginals and a copula, the dependence structure on
# the unit square. The copula is inferred from pseudo-observations, the
# ranks rescaled to (0, 1), so the fit is invariant to the marginal shapes
# and isolates the dependence. Each family's log-density is a scalar
# expression in the copula parameter and the (transformed) pseudo-
# observations, so it compiles to the existing DSL with no engine change;
# everything that depends only on the data is precomputed in R and passed as
# data columns, keeping the DSL expression shallow.
#
# Four families cover the qualitative range of bivariate dependence: Gumbel
# (upper-tail), Clayton (lower-tail), Frank (symmetric, tail-independent)
# and Gaussian (elliptical, tail-independent). The first three are
# Archimedean and parametrised for positive association in this release;
# the Gaussian family carries the sign of the dependence through its
# correlation. Family auto-selection is part of the workflow: `family =
# "auto"` fits every candidate and ranks them by predictive comparison,
# because the choice of family is a hypothesis about tail behaviour and
# belongs with the fit.

# Per-family specification. Each entry gives the unconstrained parameter to
# natural-parameter map, the DSL log-density formula, the data-column
# preparation from the pseudo-observations, the prior on the raw parameter,
# and the closed-form dependence summaries. The raw parameter is always
# `z` on the real line; the natural parameter (theta or rho) is a smooth
# transform of it, so the sampler works unconstrained.
.copula_family <- function(name) {
  specs <- list(
    gumbel = list(
      label = "gumbel",
      # theta = 1 + exp(z) >= 1; theta = 1 is independence.
      to_theta = function(z) 1 + exp(z),
      # tu = -log u, tv = -log v; A = tu^theta + tv^theta; w = A^(1/theta).
      # log c = -w + (tu + tv) + (theta - 1)(log tu + log tv)
      #         + (1/theta - 2) log A + log(w + theta - 1).
      prep = function(u, v) {
        tu <- -log(u); tv <- -log(v)
        list(tu = tu, tv = tv, ltu = log(tu), ltv = log(tv),
             tsum = tu + tv, lsum = log(tu) + log(tv))
      },
      loglik = ~ -exp(log(exp((1 + exp(z)) * ltu) +
                            exp((1 + exp(z)) * ltv)) / (1 + exp(z))) +
        tsum + exp(z) * lsum +
        (1 / (1 + exp(z)) - 2) *
          log(exp((1 + exp(z)) * ltu) + exp((1 + exp(z)) * ltv)) +
        log(exp(log(exp((1 + exp(z)) * ltu) +
                      exp((1 + exp(z)) * ltv)) / (1 + exp(z))) + exp(z)),
      data = c("tu", "tv", "ltu", "ltv", "tsum", "lsum"),
      tau = function(theta) (theta - 1) / theta,
      tail_lower = function(theta) 0,
      tail_upper = function(theta) 2 - 2^(1 / theta)
    ),
    clayton = list(
      label = "clayton",
      # theta = exp(z) > 0; theta -> 0 is independence.
      to_theta = function(z) exp(z),
      # log c = log(1 + theta) - (1 + theta)(log u + log v)
      #         - (2 + 1/theta) log(u^-theta + v^-theta - 1).
      prep = function(u, v) list(lu = log(u), lv = log(v)),
      loglik = ~ log(1 + exp(z)) - (1 + exp(z)) * (lu + lv) -
        (2 + exp(-z)) *
          log(exp(-exp(z) * lu) + exp(-exp(z) * lv) - 1),
      data = c("lu", "lv"),
      tau = function(theta) theta / (theta + 2),
      tail_lower = function(theta) 2^(-1 / theta),
      tail_upper = function(theta) 0
    ),
    frank = list(
      label = "frank",
      # theta = exp(z) > 0 (positive-association cut).
      to_theta = function(z) exp(z),
      # d = 1 - e^-theta;
      # log c = log theta + log d - theta(u + v)
      #         - 2 log(d - (1 - e^-theta u)(1 - e^-theta v)).
      prep = function(u, v) list(u = u, v = v),
      loglik = ~ z + log(1 - exp(-exp(z))) - exp(z) * (u + v) -
        2 * log((1 - exp(-exp(z))) -
                  (1 - exp(-exp(z) * u)) * (1 - exp(-exp(z) * v))),
      data = c("u", "v"),
      tau = function(theta) .frank_tau(theta),
      tail_lower = function(theta) 0,
      tail_upper = function(theta) 0
    ),
    gaussian = list(
      label = "gaussian",
      # rho = z / sqrt(1 + z^2) in (-1, 1); carries the sign of dependence.
      to_theta = function(z) z / sqrt(1 + z^2),
      # With normal scores x, y and rho = z/sqrt(1+z^2), the log-density
      # simplifies to 0.5 log(1+z^2) - 0.5 z^2 (x^2+y^2) + z sqrt(1+z^2) xy.
      prep = function(u, v) {
        x <- stats::qnorm(u); y <- stats::qnorm(v)
        list(xx = x^2 + y^2, xy = x * y)
      },
      loglik = ~ 0.5 * log(1 + z^2) - 0.5 * z^2 * xx +
        z * sqrt(1 + z^2) * xy,
      data = c("xx", "xy"),
      tau = function(rho) (2 / pi) * asin(rho),
      tail_lower = function(rho) 0,
      tail_upper = function(rho) 0
    )
  )
  specs[[name]]
}

.copula_families <- c("gumbel", "clayton", "frank", "gaussian")

# Kendall's tau of the Frank copula has no elementary closed form; it is
# 1 - 4/theta (1 - D1(theta)) with D1 the first Debye function. Computed by
# quadrature, vectorised over theta.
.frank_tau <- function(theta) {
  vapply(theta, function(th) {
    if (abs(th) < 1e-8) return(0)
    d1 <- stats::integrate(function(t) t / (exp(t) - 1), 0, th)$value / th
    1 - 4 / th * (1 - d1)
  }, numeric(1))
}

# Pseudo-observations: average ranks rescaled to the open unit square. The
# average-rank convention handles ties; heavy ties break Sklar uniqueness
# (Genest and Neslehova 2007), a caveat documented on gpum_copula.
.pseudo_obs <- function(x) rank(x, ties.method = "average") / (length(x) + 1)

#' Fit a bivariate copula to two-column data
#'
#' Infers the dependence structure of two variables by fitting a bivariate
#' copula to their pseudo-observations (average ranks rescaled to the open
#' unit square), so the fit is invariant to the marginal distributions and
#' isolates the dependence. The copula parameter is sampled with
#' [gpu_metropolis()]; the copula log-density is a normalised density on the
#' unit square, so WAIC and PSIS-LOO compare families directly.
#'
#' With `family = "auto"` every candidate family is fit and the workflow
#' returns the one preferred by predictive comparison together with the full
#' ranking, since the choice of family is a hypothesis about tail behaviour.
#' Gumbel carries upper-tail dependence, Clayton lower-tail, Frank symmetric
#' and tail-independent, Gaussian elliptical and tail-independent; a
#' selected Gumbel is itself the statement that co-movement concentrates in
#' the upper tail.
#'
#' The three Archimedean families (Gumbel, Clayton, Frank) are parametrised
#' for positive association in this release; the Gaussian family carries the
#' sign of the dependence through its correlation. For data with negative
#' rank correlation, the Gaussian family applies directly and the others do
#' not.
#'
#' @param data A two-column data frame or matrix; the columns are the two
#'   variables whose dependence is modelled.
#' @param family One of `"gumbel"`, `"clayton"`, `"frank"`, `"gaussian"`, or
#'   `"auto"` to fit all four and select by predictive comparison. Default
#'   `"auto"`.
#' @param criterion Selection criterion for `family = "auto"`: `"waic"`
#'   (default) or `"loo"`. Lower expected log predictive density loss wins.
#' @param method Sampler passed to [gpu_metropolis()]. Default `"mala"`: the
#'   copula log-density is smooth and one-dimensional in the raw parameter,
#'   and the Frank family in particular has a flat far tail in which a
#'   random walk can strand a chain, which the gradient path avoids.
#' @param warmup Warmup passed to [gpu_metropolis()]. Default `"auto"`.
#' @param n_iter,n_chains,seed,... Passed to [gpu_metropolis()].
#'
#' @return An object of class `gpum_copula`: a list with the selected
#'   `family`, the `gpum_fit`, the posterior summaries of the dependence
#'   parameter and of Kendall's tau and the tail-dependence coefficients,
#'   the pseudo-observations, and (for `"auto"`) the `comparison` table.
#'
#' @examples
#' set.seed(1)
#' u <- runif(300)
#' v <- pnorm(0.8 * qnorm(u) + 0.6 * rnorm(300))
#' fit <- gpum_copula(data.frame(x = qexp(u), y = qgamma(v, 2)),
#'                    family = "auto", n_iter = 4000, n_chains = 4)
#' fit
#'
#' @seealso [gpu_metropolis()], [kendall_tau()], [tail_dependence()]
#' @export
gpum_copula <- function(data, family = "auto", criterion = c("waic", "loo"),
                        method = "mala", warmup = "auto",
                        n_iter = 8000L, n_chains = 4L, seed = 1L, ...) {
  criterion <- match.arg(criterion)
  mat <- as.matrix(data)
  if (ncol(mat) != 2L) {
    stop("`data` must have exactly two columns.", call. = FALSE)
  }
  if (nrow(mat) < 5L) {
    stop("`data` needs at least five rows.", call. = FALSE)
  }
  u <- .pseudo_obs(mat[, 1L])
  v <- .pseudo_obs(mat[, 2L])

  if (identical(family, "auto")) {
    fits <- lapply(.copula_families, function(fam) {
      tryCatch(.copula_fit_one(fam, u, v, method, warmup, n_iter, n_chains,
                               seed, ...),
               error = function(e) NULL)
    })
    names(fits) <- .copula_families
    fits <- fits[!vapply(fits, is.null, logical(1))]
    if (!length(fits)) {
      stop("no copula family could be fit to this data.", call. = FALSE)
    }
    score <- vapply(fits, function(f) {
      val <- tryCatch(if (criterion == "waic") {
        gpum_waic(f$fit, data = f$cop_data)$waic
      } else {
        gpum_loo(f$fit, data = f$cop_data)$estimates["looic", "Estimate"]
      }, error = function(e) NA_real_)
      # A family whose chains did not converge cannot be scored fairly; its
      # comparison value is withheld rather than trusted.
      if (is.finite(f$rhat_max) && f$rhat_max >= 1.1) NA_real_ else as.numeric(val)
    }, numeric(1))
    ok <- is.finite(score)
    if (!any(ok)) {
      stop("no copula family converged well enough to be selected.",
           call. = FALSE)
    }
    ord <- order(score)
    comparison <- data.frame(
      family = names(fits)[ord],
      criterion = toupper(criterion),
      value = score[ord],
      delta = score[ord] - min(score[ok]),
      converged = is.finite(score[ord]),
      row.names = NULL
    )
    best <- fits[[which.min(score)]]
    best$comparison <- comparison
    best$criterion <- criterion
    return(best)
  }

  family <- match.arg(family, .copula_families)
  .copula_fit_one(family, u, v, method, warmup, n_iter, n_chains, seed, ...)
}

# Fit one named family to the pseudo-observations and assemble the
# gpum_copula object. The dependence summaries are posterior distributions,
# computed by pushing the parameter draws through the closed-form maps.
.copula_fit_one <- function(family, u, v, method, warmup, n_iter, n_chains,
                            seed, ...) {
  spec <- .copula_family(family)
  cop_data <- spec$prep(u, v)
  model <- gpum_model(spec$loglik, params = "z", data = spec$data,
                      prior = ~ -0.5 * (z / 2)^2)
  fit <- gpu_metropolis(model, data = cop_data, n_iter = n_iter,
                        n_chains = n_chains, method = method, warmup = warmup,
                        seed = seed, ...)
  z <- as.vector(fit$draws[, , "z"])
  theta <- spec$to_theta(z)
  # Goodness-of-fit signal: the empirical Kendall tau of the data against
  # the posterior of the model-implied tau. A well-fit family brackets the
  # empirical value; a family whose tau posterior misses it is the wrong
  # dependence shape, not just a wrong parameter.
  tau_emp <- stats::cor(u, v, method = "kendall")
  tau_post <- spec$tau(theta)
  rhat_max <- max(vapply(seq_len(fit$n_params), function(j)
    rhat(fit$draws[, , j]), numeric(1)))
  out <- list(
    family = family,
    fit = fit,
    model = model,
    cop_data = cop_data,
    pseudo_obs = cbind(u = u, v = v),
    param = .summ(theta),
    param_name = if (family == "gaussian") "rho" else "theta",
    tau = .summ(tau_post),
    tau_empirical = tau_emp,
    tail_lower = .summ(spec$tail_lower(theta)),
    tail_upper = .summ(spec$tail_upper(theta)),
    rhat_max = rhat_max,
    spec = spec
  )
  class(out) <- "gpum_copula"
  out
}

# Posterior summary of a scalar functional: mean and central 95% interval.
.summ <- function(x) {
  q <- stats::quantile(x, c(0.025, 0.5, 0.975), names = FALSE)
  c(mean = mean(x), q2.5 = q[1L], median = q[2L], q97.5 = q[3L])
}

#' @export
print.gpum_copula <- function(x, ...) {
  cat("<gpum_copula>\n")
  if (!is.null(x$comparison)) {
    cat(sprintf("  selected family : %s (by %s)\n", x$family,
                toupper(x$criterion)))
  } else {
    cat(sprintf("  family          : %s\n", x$family))
  }
  cat(sprintf("  %-15s : %.3f [%.3f, %.3f]\n", x$param_name,
              x$param["mean"], x$param["q2.5"], x$param["q97.5"]))
  cat(sprintf("  Kendall's tau   : %.3f [%.3f, %.3f]  (empirical %.3f)\n",
              x$tau["mean"], x$tau["q2.5"], x$tau["q97.5"],
              x$tau_empirical))
  cat(sprintf("  tail dependence : lower %.3f, upper %.3f\n",
              x$tail_lower["mean"], x$tail_upper["mean"]))
  # Technical diagnostic verdict: convergence plus goodness of fit, the
  # empirical Kendall tau against the model-implied posterior interval.
  conv_ok <- is.finite(x$rhat_max) && x$rhat_max < 1.05
  gof_ok <- x$tau_empirical >= x$tau["q2.5"] &&
    x$tau_empirical <= x$tau["q97.5"]
  cat(sprintf("  diagnostic      : R-hat %.3f (%s); empirical tau %s the 95%% band (%s)\n",
              x$rhat_max, if (conv_ok) "converged" else "NOT converged",
              if (gof_ok) "inside" else "outside",
              if (gof_ok) "adequate fit" else "family may be misspecified"))
  if (!is.null(x$comparison)) {
    cat("  family comparison:\n")
    print(x$comparison, row.names = FALSE, digits = 4)
  }
  invisible(x)
}

#' Visual inspection of a fitted bivariate copula
#'
#' Draws a four-panel diagnostic: the pseudo-observations with the fitted
#' copula density contours over them, the posterior of the dependence
#' parameter, the posterior of Kendall's tau with the empirical value
#' marked, and the observed-against-generated check on the unit square. This
#' is the visual companion of the text diagnostic printed by the object.
#'
#' @param x A `gpum_copula` from [gpum_copula()].
#' @param n_grid Grid resolution for the density contours. Default 40.
#' @param ... Ignored.
#' @return `x`, invisibly. Called for the plot.
#' @seealso [gpum_copula()], [gpum_density_compare()]
#' @export
plot.gpum_copula <- function(x, n_grid = 40L, ...) {
  op <- graphics::par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))
  on.exit(graphics::par(op), add = TRUE)
  u <- x$pseudo_obs[, "u"]; v <- x$pseudo_obs[, "v"]
  spec <- x$spec
  theta_hat <- x$param["median"]

  # Panel 1: pseudo-obs with fitted copula density contours.
  gr <- seq(0.02, 0.98, length.out = n_grid)
  gg <- expand.grid(u = gr, v = gr)
  cd <- spec$prep(gg$u, gg$v)
  ll <- .gpum_copula_logdens(spec, cd, x)
  dv <- exp(ll)
  # Clamp non-finite density on the grid edges (the Archimedean densities
  # diverge at the corners) so the contour routine has a clean surface.
  dv[!is.finite(dv)] <- NA_real_
  cap <- stats::quantile(dv, 0.99, na.rm = TRUE)
  dv[!is.na(dv) & dv > cap] <- cap
  dens <- matrix(dv, n_grid, n_grid)
  graphics::plot(u, v, pch = 16, col = grDevices::adjustcolor("black", 0.35),
                 cex = 0.6, xlab = "u", ylab = "v",
                 main = sprintf("pseudo-obs + %s density", x$family))
  graphics::contour(gr, gr, dens, add = TRUE, col = "steelblue", lwd = 1.2)

  # Panel 2: posterior of the dependence parameter.
  z <- as.vector(x$fit$draws[, , "z"])
  theta <- spec$to_theta(z)
  graphics::hist(theta, breaks = 40, col = "grey85", border = "white",
                 main = sprintf("posterior of %s", x$param_name),
                 xlab = x$param_name, freq = FALSE)
  graphics::abline(v = x$param["median"], col = "steelblue", lwd = 2)

  # Panel 3: posterior of Kendall's tau, empirical value marked.
  tau <- spec$tau(theta)
  graphics::hist(tau, breaks = 40, col = "grey85", border = "white",
                 main = "Kendall's tau: posterior vs empirical",
                 xlab = "tau", freq = FALSE)
  graphics::abline(v = x$tau_empirical, col = "firebrick", lwd = 2, lty = 2)
  graphics::abline(v = x$tau["median"], col = "steelblue", lwd = 2)
  graphics::legend("topright", bty = "n", lwd = 2, lty = c(1, 2),
                   col = c("steelblue", "firebrick"),
                   legend = c("model", "empirical"), cex = 0.8)

  # Panel 4: observed vs generated pseudo-obs on the unit square.
  gen <- .copula_generate(x, nrow(x$pseudo_obs))
  graphics::plot(u, v, pch = 16, col = grDevices::adjustcolor("black", 0.35),
                 cex = 0.6, xlab = "u", ylab = "v",
                 main = "observed (black) vs generated (blue)")
  graphics::points(gen[, 1L], gen[, 2L], pch = 16, cex = 0.6,
                   col = grDevices::adjustcolor("steelblue", 0.35))
  invisible(x)
}

# Log copula density per grid cell at the posterior median parameter,
# reusing the compiled model so the contours match the fit. The pointwise
# evaluator returns one value per data row (grid cell), not the row sum.
.gpum_copula_logdens <- function(spec, cop_data, x) {
  z_hat <- stats::median(as.vector(x$fit$draws[, , "z"]))
  d <- .gpum_data_flat(x$model, cop_data)
  as.numeric(rust_loglik_pointwise(x$model$loglik$code, x$model$loglik$consts,
                                   1L, as.numeric(d$flat), x$model$n_cols,
                                   d$n_obs, z_hat))
}

# Generate pseudo-observations from the fitted copula by conditional
# sampling: draw u uniform, then v from the conditional copula C(v | u) at
# the posterior median parameter via numeric inversion. This backs the
# observed-vs-generated visual check.
.copula_generate <- function(x, n) {
  spec <- x$spec
  z_hat <- stats::median(as.vector(x$fit$draws[, , "z"]))
  theta <- spec$to_theta(z_hat)
  u <- stats::runif(n)
  w <- stats::runif(n)
  v <- .copula_cond_inv(x$family, u, w, theta)
  cbind(u = u, v = v)
}

# Inverse of the conditional copula h(v | u) = dC/du, by bisection on the
# unit interval. Family-specific h-functions; stable and vectorised.
.copula_cond_inv <- function(family, u, w, theta) {
  if (family == "clayton") {
    # Closed-form conditional inverse for Clayton.
    return(((w^(-theta / (1 + theta)) - 1) * u^(-theta) + 1)^(-1 / theta))
  }
  hfun <- switch(family,
    gumbel = function(v) .h_gumbel(u, v, theta),
    frank = function(v) .h_frank(u, v, theta),
    gaussian = function(v) stats::pnorm(
      (stats::qnorm(v) - theta * stats::qnorm(u)) / sqrt(1 - theta^2))
  )
  lo <- rep(1e-6, length(u)); hi <- rep(1 - 1e-6, length(u))
  for (i in seq_len(40L)) {
    mid <- (lo + hi) / 2
    h <- hfun(mid)
    hi <- ifelse(h > w, mid, hi)
    lo <- ifelse(h > w, lo, mid)
  }
  (lo + hi) / 2
}

# Conditional copula h(v|u) = dC/du for Gumbel and Frank.
.h_gumbel <- function(u, v, theta) {
  tu <- -log(u); tv <- -log(v)
  A <- tu^theta + tv^theta
  w <- A^(1 / theta)
  exp(-w) * (1 / u) * tu^(theta - 1) * A^(1 / theta - 1)
}
.h_frank <- function(u, v, theta) {
  num <- exp(-theta * u) * (exp(-theta * v) - 1)
  den <- (exp(-theta) - 1) + (exp(-theta * u) - 1) * (exp(-theta * v) - 1)
  num / den
}

#' Posterior Kendall's tau of a fitted copula
#'
#' Returns the posterior summary of Kendall's tau, the rank-correlation
#' functional of the copula parameter, on its natural `[-1, 1]` scale. Tau
#' is comparable across families, unlike the family-specific dependence
#' parameter.
#'
#' @param object A `gpum_copula` from [gpum_copula()].
#' @return A named numeric vector: `mean`, `q2.5`, `median`, `q97.5`.
#' @seealso [gpum_copula()], [tail_dependence()]
#' @export
kendall_tau <- function(object) {
  if (!inherits(object, "gpum_copula")) {
    stop("`object` must come from gpum_copula().", call. = FALSE)
  }
  object$tau
}

#' Posterior tail-dependence coefficients of a fitted copula
#'
#' Returns the posterior summaries of the lower and upper tail-dependence
#' coefficients, the probabilities of joint extremes in each tail. Gumbel
#' has upper-tail dependence and no lower, Clayton the reverse, Frank and
#' Gaussian neither.
#'
#' @param object A `gpum_copula` from [gpum_copula()].
#' @return A list with `lower` and `upper`, each a named numeric vector.
#' @seealso [gpum_copula()], [kendall_tau()]
#' @export
tail_dependence <- function(object) {
  if (!inherits(object, "gpum_copula")) {
    stop("`object` must come from gpum_copula().", call. = FALSE)
  }
  list(lower = object$tail_lower, upper = object$tail_upper)
}
