# The conjugate fast path. A Gaussian linear model with a
# Normal-inverse-Gamma prior has its joint posterior in closed form, so no
# Markov chain is needed at all: the draws are independent samples from the
# exact posterior. This is the regime where a Gibbs specialist outruns any
# Metropolis sampler; the honest answer is not a faster random walk but the
# closed form itself, which is one step better than Gibbs (independent draws
# rather than a conditional-update chain).

#' Declare a conjugate Gaussian linear model
#'
#' Builds the conjugate specification of the Gaussian linear model
#' `y = X beta + eps`, `eps ~ N(0, sigma^2 I)`, under the
#' Normal-inverse-Gamma prior
#' `beta | sigma^2 ~ N(b0, sigma^2 B0^{-1})`,
#' `sigma^2 ~ InvGamma(c0 / 2, d0 / 2)`,
#' with `B0` a prior precision matrix. Under this prior the joint posterior
#' of `(beta, sigma^2)` is again Normal-inverse-Gamma in closed form, so
#' [gpu_metropolis()] samples it exactly and independently, with no warmup,
#' no proposal and no rejection; the effective sample size equals the number
#' of draws. The marginal likelihood is also available in closed form and is
#' returned by the fit as `log_marginal`.
#'
#' The prior must be proper (`B0` positive definite, `c0 > 0`, `d0 > 0`) for
#' the marginal likelihood to exist; the sampling itself only needs the
#' posterior precision `X'X + B0` to be positive definite. The defaults are
#' weakly informative and proper.
#'
#' @param formula A standard R model formula, e.g. `mpg ~ wt`. Interactions
#'   and transformations follow the usual `model.matrix` rules.
#' @param data A data frame holding the variables of `formula`.
#' @param b0 Prior mean of `beta`; scalar recycled or a vector of length
#'   equal to the number of columns of the design matrix. Default 0.
#' @param B0 Prior precision of `beta` (inverse covariance, scaled by
#'   `sigma^2`); a scalar for `B0 * I` or a full matrix. Default `1e-6`,
#'   near-flat but proper.
#' @param c0 Prior shape times two of `sigma^2`: `sigma^2 ~
#'   InvGamma(c0 / 2, d0 / 2)`. Default 0.002.
#' @param d0 Prior rate times two of `sigma^2`. Default 0.002.
#'
#' @return An object of class `gpum_conjugate` holding the design matrix,
#'   the response and the prior, ready for [gpu_metropolis()].
#'
#' @examples
#' m <- gpum_lm(mpg ~ wt, data = mtcars)
#' fit <- gpu_metropolis(m, n_iter = 4000, n_chains = 4, seed = 1)
#' fit$log_marginal
#'
#' @seealso [gpu_metropolis()], [gpum_model()]
#' @export
gpum_lm <- function(formula, data, b0 = 0, B0 = 1e-6, c0 = 0.002,
                    d0 = 0.002) {
  mf <- stats::model.frame(formula, data)
  y <- as.numeric(stats::model.response(mf))
  X <- stats::model.matrix(attr(mf, "terms"), mf)
  p <- ncol(X)
  b0 <- rep_len(as.numeric(b0), p)
  B0 <- if (is.matrix(B0)) {
    if (!all(dim(B0) == c(p, p))) {
      stop("`B0` matrix must be ", p, " by ", p, ".", call. = FALSE)
    }
    B0
  } else {
    diag(as.numeric(B0), p)
  }
  if (!isTRUE(all(eigen(B0, symmetric = TRUE,
                        only.values = TRUE)$values > 0))) {
    stop("`B0` must be positive definite (a proper prior).", call. = FALSE)
  }
  if (c0 <= 0 || d0 <= 0) {
    stop("`c0` and `d0` must be positive (a proper prior).", call. = FALSE)
  }
  structure(
    list(formula = formula, y = y, X = X, b0 = b0, B0 = B0,
         c0 = as.numeric(c0), d0 = as.numeric(d0),
         params = c(colnames(X), "sigma2"),
         n_params = p + 1L, n_obs = length(y)),
    class = "gpum_conjugate"
  )
}

#' @export
print.gpum_conjugate <- function(x, ...) {
  cat("<gpum_conjugate>\n")
  cat(sprintf("  model      : %s\n", deparse(x$formula)))
  cat(sprintf("  coefficients: %s\n",
              paste(setdiff(x$params, "sigma2"), collapse = ", ")))
  cat(sprintf("  observations: %d\n", x$n_obs))
  cat("  posterior  : Normal-inverse-Gamma, closed form (exact sampling)\n")
  invisible(x)
}

# Closed-form posterior of the conjugate Gaussian linear model, plus the log
# marginal likelihood. Precision parametrisation throughout: `Bn` is the
# posterior precision of beta (scaled by sigma^2), `cn / 2` and `dn / 2` the
# posterior shape and rate of the inverse-Gamma.
.conjugate_posterior <- function(model) {
  X <- model$X
  y <- model$y
  n <- model$n_obs
  Bn <- crossprod(X) + model$B0
  mn <- solve(Bn, crossprod(X, y) + model$B0 %*% model$b0)
  cn <- model$c0 + n
  dn <- model$d0 + sum(y^2) +
    as.numeric(t(model$b0) %*% model$B0 %*% model$b0) -
    as.numeric(t(mn) %*% Bn %*% mn)
  # log m(y) = -(n/2) log(2 pi) + (1/2) (log det B0 - log det Bn)
  #            + lgamma(cn/2) - lgamma(c0/2)
  #            + (c0/2) log(d0/2) - (cn/2) log(dn/2)
  log_marginal <- -0.5 * n * log(2 * pi) +
    0.5 * (determinant(model$B0, logarithm = TRUE)$modulus -
             determinant(Bn, logarithm = TRUE)$modulus) +
    lgamma(cn / 2) - lgamma(model$c0 / 2) +
    (model$c0 / 2) * log(model$d0 / 2) - (cn / 2) * log(dn / 2)
  list(Bn = Bn, mn = as.numeric(mn), cn = cn, dn = dn,
       log_marginal = as.numeric(log_marginal))
}

# Exact independent sampling from the Normal-inverse-Gamma posterior,
# vectorised: one Gamma vector for sigma^2, one standard normal matrix pushed
# through the Cholesky of the posterior covariance and scaled per draw.
.conjugate_sample <- function(post, p, n_draws) {
  sigma2 <- 1 / stats::rgamma(n_draws, shape = post$cn / 2,
                              rate = post$dn / 2)
  # chol() returns the upper factor R with Bn = R'R, so Bn^{-1} = R^{-1} R^{-T}
  # and beta = mn + sqrt(sigma2) * R^{-1} z has covariance sigma2 * Bn^{-1}.
  R <- chol(post$Bn)
  Z <- matrix(stats::rnorm(n_draws * p), nrow = p, ncol = n_draws)
  beta <- post$mn + backsolve(R, Z) *
    rep(sqrt(sigma2), each = p)
  cbind(t(beta), sigma2)
}

# The gpu_metropolis() dispatch target for gpum_conjugate models. Returns a
# gpum_fit so the whole downstream toolkit (hdi, hypothesis, ROPE, pairs,
# diagnose) consumes it unchanged; accept_rate is 1 and warmup is 0 because
# nothing is proposed and nothing needs to converge.
.gpum_exact_fit <- function(model, n_iter, n_chains, seed) {
  n_iter <- as.integer(n_iter)
  n_chains <- as.integer(n_chains)
  if (n_iter < 1L || n_chains < 1L) {
    stop("`n_iter` and `n_chains` must be positive.", call. = FALSE)
  }
  post <- .conjugate_posterior(model)
  p <- model$n_params - 1L
  if (exists(".Random.seed", envir = .GlobalEnv)) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv)
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
  }
  set.seed(as.integer(seed))
  flat <- .conjugate_sample(post, p, n_iter * n_chains)
  draws <- array(NA_real_, dim = c(n_iter, n_chains, model$n_params),
                 dimnames = list(NULL, NULL, model$params))
  for (j in seq_len(model$n_params)) {
    draws[, , j] <- matrix(flat[, j], nrow = n_iter, ncol = n_chains)
  }
  structure(
    list(draws = draws, accept_rate = rep(1, n_chains), model = model,
         n_iter = n_iter, n_iter_total = n_iter, warmup = 0L,
         n_chains = n_chains, n_params = model$n_params,
         backend = "cpu", seed = seed, method = "exact",
         posterior = post, log_marginal = post$log_marginal),
    class = "gpum_fit"
  )
}
