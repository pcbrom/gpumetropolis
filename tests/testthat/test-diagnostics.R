# Phase 0 verification: the distributional equivalence harness behaves as
# expected on converged, non-converged, equivalent and non-equivalent inputs.

test_that("split R-hat is near 1 for converged chains", {
  set.seed(1)
  x <- rnorm(1500, mean = 0, sd = 1)
  fit <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 4000, n_chains = 4,
                                  seed = 21)
  expect_lt(rhat(fit), 1.05)
})

test_that("split R-hat flags chains stuck at different locations", {
  set.seed(2)
  # Two chains pinned far apart with negligible spread: no mixing.
  stuck <- cbind(rnorm(400, mean = 0, sd = 0.1),
                 rnorm(400, mean = 10, sd = 0.1))
  expect_gt(rhat(stuck, warmup = 0), 1.5)
})

test_that("split R-hat needs enough post-warmup iterations", {
  expect_error(rhat(matrix(1:6, nrow = 3, ncol = 2)))
})

test_that("ks_equivalence does not flag two runs of the same target", {
  set.seed(3)
  x <- rnorm(800, mean = 1, sd = 1)
  a <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 4000, seed = 1)
  b <- metropolis_gaussian_mean(x, sigma = 1, n_iter = 4000, seed = 2)
  # Both runs are converged before the test relies on equivalence.
  expect_lt(rhat(a), 1.05)
  expect_lt(rhat(b), 1.05)
  res <- ks_equivalence(a, b)
  expect_true(res$equivalent)
  expect_gt(res$p_value, 0.05)
  expect_named(res, c("statistic", "p_value", "alpha", "equivalent",
                      "n_x", "n_y"))
})

test_that("ks_equivalence flags clearly different posteriors", {
  set.seed(4)
  xa <- rnorm(500, mean = 0, sd = 1)
  xb <- rnorm(500, mean = 5, sd = 1)
  a <- metropolis_gaussian_mean(xa, sigma = 1, n_iter = 2000, seed = 1)
  b <- metropolis_gaussian_mean(xb, sigma = 1, n_iter = 2000, seed = 1)
  res <- ks_equivalence(a, b)
  expect_false(res$equivalent)
  expect_lt(res$p_value, 0.05)
})

test_that("ks_equivalence accepts matrices and vectors", {
  set.seed(5)
  v1 <- rnorm(2000)
  v2 <- rnorm(2000)
  res <- ks_equivalence(v1, v2, thin = FALSE)
  expect_type(res$equivalent, "logical")
  expect_equal(res$n_x, 2000L)
})

test_that("ess of independent draws is close to their count", {
  set.seed(8)
  v <- rnorm(4000)
  e <- ess(v)
  expect_gt(e, 3000)
  expect_lte(e, 4000)
})

test_that("gaussian_mean_posterior returns the closed-form posterior", {
  set.seed(6)
  x <- rnorm(1000, mean = 2, sd = 4)
  post <- gaussian_mean_posterior(x, sigma = 4)
  expect_equal(post$mean, mean(x))
  expect_equal(post$sd, 4 / sqrt(1000))
})

test_that("diagnostics reject malformed input", {
  expect_error(rhat("not a matrix"))
  expect_error(ks_equivalence(rnorm(10), rnorm(10), alpha = 1.5))
  expect_error(gaussian_mean_posterior(numeric(0), sigma = 1))
})
