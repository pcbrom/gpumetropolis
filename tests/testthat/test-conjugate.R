test_that("gpum_lm builds the conjugate spec from a formula", {
  m <- gpum_lm(mpg ~ wt, data = mtcars)
  expect_s3_class(m, "gpum_conjugate")
  expect_equal(m$params, c("(Intercept)", "wt", "sigma2"))
  expect_equal(m$n_params, 3L)
  expect_equal(m$n_obs, 32L)
  expect_output(print(m), "closed form")
})

test_that("gpum_lm rejects improper priors", {
  expect_error(gpum_lm(mpg ~ wt, data = mtcars, B0 = matrix(0, 2, 2)),
               "positive definite")
  expect_error(gpum_lm(mpg ~ wt, data = mtcars, c0 = 0), "positive")
  expect_error(gpum_lm(mpg ~ wt, data = mtcars, B0 = diag(1, 3)),
               "must be 2 by 2")
})

test_that("the exact fit reproduces the closed-form posterior and lm", {
  m <- gpum_lm(mpg ~ wt, data = mtcars)
  fit <- gpu_metropolis(m, n_iter = 50000, n_chains = 4, seed = 1)
  expect_s3_class(fit, "gpum_fit")
  expect_equal(fit$method, "exact")
  expect_equal(fit$warmup, 0L)
  expect_equal(fit$accept_rate, rep(1, 4))
  # Monte Carlo means against the analytic posterior mean.
  post <- .conjugate_posterior(m)
  expect_equal(mean(fit$draws[, , "(Intercept)"]), post$mn[1],
               tolerance = 0.01)
  expect_equal(mean(fit$draws[, , "wt"]), post$mn[2], tolerance = 0.01)
  # Under the near-flat default prior the posterior mean matches lm() and
  # the posterior sd of beta matches the lm standard errors.
  cf <- stats::coef(summary(stats::lm(mpg ~ wt, data = mtcars)))
  expect_equal(mean(fit$draws[, , "wt"]), cf["wt", 1], tolerance = 0.01)
  expect_equal(stats::sd(fit$draws[, , "wt"]), cf["wt", 2],
               tolerance = 0.02)
  # sigma2 draws against the analytic inverse-Gamma mean dn / (cn - 2).
  expect_equal(mean(fit$draws[, , "sigma2"]), post$dn / (post$cn - 2),
               tolerance = 0.02)
})

test_that("the exact fit is reproducible from the seed and iid across draws", {
  m <- gpum_lm(mpg ~ wt, data = mtcars)
  f1 <- gpu_metropolis(m, n_iter = 500, n_chains = 2, seed = 7)
  f2 <- gpu_metropolis(m, n_iter = 500, n_chains = 2, seed = 7)
  expect_identical(f1$draws, f2$draws)
  f3 <- gpu_metropolis(m, n_iter = 500, n_chains = 2, seed = 8)
  expect_false(identical(f1$draws, f3$draws))
  # Independent draws: lag-1 autocorrelation is noise-level.
  ac <- stats::acf(f1$draws[, 1, "wt"], lag.max = 1, plot = FALSE)$acf[2]
  expect_lt(abs(ac), 0.15)
})

test_that("the closed-form log marginal matches numeric double integration", {
  set.seed(2)
  yv <- stats::rnorm(12, 1.5, 0.8)
  m1 <- gpum_lm(yv ~ 1, data = data.frame(yv = yv), B0 = 1, c0 = 2, d0 = 2)
  p1 <- .conjugate_posterior(m1)
  # Integrate the likelihood times the Normal-inverse-Gamma prior in
  # u = log(sigma2), so the heavy inverse-Gamma tail is covered.
  integrand <- function(mu, u) {
    s2 <- exp(u)
    exp(sum(stats::dnorm(yv, mu, sqrt(s2), log = TRUE)) +
          stats::dnorm(mu, 0, sqrt(s2), log = TRUE) +
          (-2) * log(s2) - 1 / s2 + u)
  }
  num <- stats::integrate(function(uv) sapply(uv, function(u)
    stats::integrate(function(muv) sapply(muv, integrand, u = u),
                     -15, 15, rel.tol = 1e-10)$value),
    log(1e-5), log(1e6), rel.tol = 1e-9)$value
  expect_equal(p1$log_marginal, log(num), tolerance = 1e-3)
})

test_that("the exact fit flows through the downstream toolkit", {
  m <- gpum_lm(mpg ~ wt, data = mtcars)
  fit <- gpu_metropolis(m, n_iter = 4000, n_chains = 4, seed = 1)
  expect_output(print(fit), "exact")
  h <- gpum_hypothesis(fit, "wt", lower = -Inf, upper = 0)
  expect_gt(h$prob, 0.99)
  ci <- hdi(as.vector(fit$draws[, , "wt"]))
  expect_lt(ci[1], -5.34)
  expect_gt(ci[2], -5.35)
  expect_true(is.finite(fit$log_marginal))
})

test_that("gpu_metropolis still rejects objects that are neither model kind", {
  expect_error(gpu_metropolis(list()), "gpum_model")
})
