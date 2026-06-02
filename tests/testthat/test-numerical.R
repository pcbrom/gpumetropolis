# Numerical edge cases: the sampler must degrade gracefully when the
# log-density can become non-finite, not crash and not corrupt the draws.

test_that("a proposal that makes the log-density non-finite is rejected", {
  # Target proportional to mu * exp(-mu), defined for mu > 0. A proposal with
  # mu <= 0 makes log(mu) non-finite; the step must be rejected, not crash.
  set.seed(1)
  m <- gpum_model(~ log(mu) - mu + 0 * y, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = 0), init = matrix(1.5, 1, 1),
                        proposal_sd = 0.4, n_iter = 3000, seed = 1,
                        backend = "cpu")
  draws <- fit$draws[, , 1]
  expect_true(all(is.finite(draws)))
  # The chain stays in the valid region; the posterior mean of Gamma(2,1) is 2.
  expect_gt(min(draws), 0)
  expect_equal(mean(draws), 2, tolerance = 0.4)
})

test_that("a large argument to exp does not crash the sampler", {
  # exp of the parameter can overflow to infinity for large mu; the run must
  # complete and return finite draws.
  set.seed(2)
  m <- gpum_model(~ -exp(mu) - mu + 0 * y, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = 0), init = matrix(0, 1, 1),
                        proposal_sd = 0.5, n_iter = 2000, seed = 2,
                        backend = "cpu")
  expect_true(all(is.finite(fit$draws[, , 1])))
})

test_that("a non-finite starting log-density does not crash the run", {
  # Starting outside the valid region (mu <= 0 with a log(mu) term) makes the
  # initial log-density non-finite. The run must still complete without error.
  m <- gpum_model(~ log(mu) - mu + 0 * y, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = 0), init = matrix(-1, 1, 1),
                        proposal_sd = 0.3, n_iter = 500, seed = 1,
                        backend = "cpu")
  expect_s3_class(fit, "gpum_fit")
  expect_equal(dim(fit$draws), c(250L, 1L, 1L))
})
