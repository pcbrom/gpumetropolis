# Differential Evolution MCMC integration tests. The DE proposal builds each
# step from the scaled difference of two other chains, so it aligns with the
# correlation of the target without an explicit covariance. The strong test is
# a closed-form correlated bivariate posterior, where a diagonal random-walk
# proposal mixes slowly and the difference proposal should not.

test_that("method = \"de\" needs at least four chains", {
  set.seed(1)
  y <- rnorm(200, mean = 1, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  expect_error(
    gpu_metropolis(m, data = list(y = y), n_iter = 100, n_chains = 3,
                   method = "de", backend = "cpu"),
    "at least 4 chains"
  )
})

test_that("the default gamma is 2.38 / sqrt(2 d)", {
  set.seed(1)
  y <- rnorm(200, mean = 1, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 400, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  expect_identical(fit$method, "de")
  expect_equal(fit$gamma, 2.38 / sqrt(2 * 1), tolerance = 1e-9)
})

test_that("DE recovers the Gaussian-mean posterior", {
  set.seed(1)
  y <- rnorm(4000, mean = 3.4, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 4000, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  draws <- fit$draws[, , "mu"]
  # Closed-form posterior with a flat prior and sigma = 2.
  post_mean <- mean(y)
  post_sd <- 2 / sqrt(length(y))
  expect_lt(abs(mean(draws) - post_mean), 0.02)
  expect_lt(abs(stats::sd(as.vector(draws)) - post_sd), 0.01)
  expect_lt(rhat(draws, warmup = 0), 1.05)
})

test_that("DE recovers a strongly correlated bivariate posterior", {
  set.seed(1)
  rho <- 0.9
  Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
  n <- 500
  L <- chol(Sigma)
  y <- matrix(rnorm(n * 2), n, 2) %*% L
  y[, 1] <- y[, 1] + 2
  y[, 2] <- y[, 2] - 1
  ybar <- colMeans(y)
  post_sd <- sqrt(diag(Sigma) / n)

  Si <- solve(Sigma)
  loglik <- stats::as.formula(sprintf(
    "~ -0.5 * (%.6f*(y1-mu1)^2 + 2*%.6f*(y1-mu1)*(y2-mu2) + %.6f*(y2-mu2)^2)",
    Si[1, 1], Si[1, 2], Si[2, 2]))
  m <- gpum_model(loglik, params = c("mu1", "mu2"), data = c("y1", "y2"))

  fit <- gpu_metropolis(m, data = list(y1 = y[, 1], y2 = y[, 2]),
                        n_iter = 8000, n_chains = 16, method = "de",
                        seed = 1, backend = "cpu")
  d1 <- as.vector(fit$draws[, , "mu1"])
  d2 <- as.vector(fit$draws[, , "mu2"])
  expect_lt(abs(mean(d1) - ybar[1]), 0.02)
  expect_lt(abs(mean(d2) - ybar[2]), 0.02)
  expect_lt(abs(stats::sd(d1) - post_sd[1]), 0.01)
  # The defining check: the difference proposal recovers the correlation.
  expect_lt(abs(stats::cor(d1, d2) - rho), 0.05)
  expect_lt(rhat(fit$draws[, , "mu1"], warmup = 0), 1.05)
})

test_that(".de_orchestrate is reproducible from the seed", {
  set.seed(1)
  y <- rnorm(300, mean = 1, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  f1 <- gpu_metropolis(m, data = list(y = y), n_iter = 600, n_chains = 8,
                       method = "de", seed = 7, backend = "cpu")
  f2 <- gpu_metropolis(m, data = list(y = y), n_iter = 600, n_chains = 8,
                       method = "de", seed = 7, backend = "cpu")
  expect_identical(f1$draws, f2$draws)
})

test_that("gpum_diagnose recognises a DE fit", {
  set.seed(1)
  y <- rnorm(500, mean = 2, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 2000, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  res <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_true("de_hint" %in% names(res))
  expect_false(is.null(fit$adaptation$disp_history))
})
