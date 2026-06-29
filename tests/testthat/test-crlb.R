# Tests for the optional Cramer-Rao diagnostic. On a regular Gaussian model
# the bound is known in closed form (Sigma / n), so gpum_crlb must recover it;
# on the irregular cases (no data, unconverged) it must refuse to report a
# number and say why.

test_that("gpum_crlb recovers the closed-form bound on a correlated Gaussian", {
  set.seed(1)
  rho <- 0.9
  Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
  n <- 500
  L <- chol(Sigma)
  y <- matrix(rnorm(n * 2), n, 2) %*% L
  y[, 1] <- y[, 1] + 2
  y[, 2] <- y[, 2] - 1
  Si <- solve(Sigma)
  loglik <- stats::as.formula(sprintf(
    "~ -0.5 * (%.6f*(y1-mu1)^2 + 2*%.6f*(y1-mu1)*(y2-mu2) + %.6f*(y2-mu2)^2)",
    Si[1, 1], Si[1, 2], Si[2, 2]))
  m <- gpum_model(loglik, params = c("mu1", "mu2"), data = c("y1", "y2"))
  dat <- list(y1 = y[, 1], y2 = y[, 2])
  fit <- gpu_metropolis(m, data = dat, n_iter = 8000, n_chains = 16,
                        method = "de", seed = 1, backend = "cpu")

  cr <- gpum_crlb(fit, data = dat)
  expect_true(cr$applicable)
  # Analytic bound is Sigma / n.
  analytic_sd <- sqrt(diag(Sigma) / n)
  expect_lt(max(abs(cr$crlb_sd - analytic_sd)), 0.005)
  expect_lt(abs(cr$crlb_cor[1, 2] - rho), 0.02)
  # The posterior spread tracks the bound (ratio near one).
  expect_lt(max(abs(cr$posterior_sd / cr$crlb_sd - 1)), 0.1)
})

test_that("gpum_crlb refuses a model with no data term", {
  banana <- gpum_model(
    loglik = ~ 0, params = c("x1", "x2"),
    prior = ~ -x1^2 / 200 - 0.5 * (x2 + 0.1 * x1^2 - 10)^2)
  set.seed(11)
  init <- cbind(rnorm(20, 0, 8), rnorm(20, 0, 4))
  fit <- gpu_metropolis(banana, init = init, n_iter = 2000, method = "de",
                        seed = 1, backend = "cpu")
  cr <- gpum_crlb(fit)
  expect_false(cr$applicable)
  expect_match(cr$note, "no data term")
  expect_true(all(is.na(cr$crlb_sd)))
})

test_that("gpum_crlb requires data for a model that has a data term", {
  set.seed(1)
  y <- rnorm(300, mean = 1, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 800, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  expect_error(gpum_crlb(fit), "data.*required")
})

test_that("gpum_crlb flags an unconverged multimodal fit as not applicable", {
  set.seed(1)
  y <- rnorm(400, mean = 3, sd = 1)
  m <- gpum_model(
    loglik = ~ log(exp(-((y - mu)^2) / 2) + exp(-((y + mu)^2) / 2)),
    params = "mu", data = "y")
  # Random-walk chains from separated starts stay stuck, giving a large R-hat.
  init <- matrix(seq(-6, 6, length.out = 8), nrow = 8, ncol = 1)
  fit <- gpu_metropolis(m, data = list(y = y), init = init,
                        proposal_sd = 0.15, n_iter = 2000, method = "rwm",
                        seed = 1, backend = "cpu")
  cr <- gpum_crlb(fit, data = list(y = y))
  expect_false(cr$applicable)
  expect_match(cr$note, "R-hat")
})

test_that("gpum_crlb prints without error", {
  set.seed(1)
  y <- rnorm(2000, mean = 3, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 4000, n_chains = 12,
                        method = "de", seed = 1, backend = "cpu")
  cr <- gpum_crlb(fit, data = list(y = y))
  expect_output(print(cr), "gpum_crlb")
})
