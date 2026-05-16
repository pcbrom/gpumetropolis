# Phase 0 verification: the CPU reference sampler recovers known parameters
# and is reproducible.

test_that("sampler returns a well formed gpumetropolis_fit", {
  set.seed(1)
  x <- rnorm(500, mean = 0, sd = 1)
  fit <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 1000, n_chains = 4,
                                  seed = 1)
  expect_s3_class(fit, "gpumetropolis_fit")
  expect_equal(dim(fit$draws), c(1000L, 4L))
  expect_length(fit$accept_rate, 4L)
  expect_equal(fit$n_chains, 4L)
  expect_equal(fit$n_iter, 1000L)
})

test_that("sampler recovers the analytic Gaussian-mean posterior", {
  set.seed(42)
  x <- rnorm(2000, mean = 5, sd = 3)
  fit <- metropolis_gaussian_mean(x, sigma = 3, n_iter = 4000, n_chains = 4,
                                  seed = 7)
  post <- gaussian_mean_posterior(x, sigma = 3)

  draws <- as.vector(fit$draws[2001:4000, ])
  expect_equal(mean(draws), post$mean, tolerance = 0.02)
  expect_equal(stats::sd(draws), post$sd, tolerance = 0.15)
})

test_that("acceptance rate sits in a sane range for the default proposal", {
  set.seed(3)
  x <- rnorm(1000, mean = 2, sd = 2)
  fit <- metropolis_gaussian_mean(x, sigma = 2, n_iter = 3000, seed = 11)
  expect_true(all(fit$accept_rate > 0.2 & fit$accept_rate < 0.7))
})

test_that("chains from overdispersed starts converge (split R-hat near 1)", {
  set.seed(5)
  x <- rnorm(1500, mean = -1, sd = 2)
  fit <- metropolis_gaussian_mean(x, sigma = 2, n_iter = 4000, n_chains = 4,
                                  seed = 13)
  expect_lt(rhat(fit), 1.05)
})

test_that("same seed reproduces draws, different seed does not", {
  set.seed(7)
  x <- rnorm(400, mean = 1, sd = 1)
  f1 <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 500, seed = 1)
  f2 <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 500, seed = 1)
  f3 <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 500, seed = 2)
  expect_identical(f1$draws, f2$draws)
  expect_false(identical(f1$draws, f3$draws))
})

test_that("init length sets the number of chains", {
  set.seed(9)
  x <- rnorm(300, mean = 0, sd = 1)
  fit <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 200,
                                  init = c(-2, 0, 2))
  expect_equal(fit$n_chains, 3L)
  expect_equal(ncol(fit$draws), 3L)
  expect_identical(colnames(fit$draws), c("chain1", "chain2", "chain3"))
})

test_that("invalid input is rejected", {
  x <- rnorm(100)
  expect_error(metropolis_gaussian_mean(numeric(0), sigma = 1))
  expect_error(metropolis_gaussian_mean(x, sigma = -1))
  expect_error(metropolis_gaussian_mean(x, sigma = 0))
  expect_error(metropolis_gaussian_mean(x, sigma = 1, n_iter = 0))
  expect_error(metropolis_gaussian_mean(x, sigma = 1, proposal_sd = -1))
  expect_error(metropolis_gaussian_mean(c(1, NA, 3), sigma = 1))
})
