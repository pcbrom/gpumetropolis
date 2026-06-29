# Tests for the formal Bayesian decision and comparison layer: posterior
# probability of a hypothesis, ROPE and HDI, WAIC and PSIS-LOO, and the
# marginal likelihood and Bayes factor by thermodynamic integration.

test_that("hdi recovers the central interval of a normal sample", {
  set.seed(1)
  x <- rnorm(50000)
  h <- hdi(x, ci = 0.95)
  expect_lt(abs(h[["lower"]] + 1.96), 0.1)
  expect_lt(abs(h[["upper"]] - 1.96), 0.1)
})

test_that("gpum_hypothesis reports posterior tail probabilities", {
  set.seed(1)
  y <- rnorm(4000, mean = 0.5, sd = 2)         # posterior mean near 0.5
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 4000, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  h <- gpum_hypothesis(fit, "mu", lower = 0)
  expect_s3_class(h, "gpum_hypothesis")
  # posterior mean ~0.5 with sd ~2/sqrt(4000) ~0.032, so P(mu>0) ~1. With
  # upper = Inf the one-sided mass is `prob`, the interval (0, Inf).
  expect_gt(h$prob, 0.95)
  expect_lt(h$prob_below, 0.05)
  expect_equal(h$prob + h$prob_below + h$prob_above, 1, tolerance = 1e-8)
})

test_that("gpum_rope decides equivalence and difference", {
  set.seed(1)
  # A posterior tightly around 3: practically different from 0.
  y <- rnorm(4000, mean = 3, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 4000, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  far <- gpum_rope(fit, "mu", rope = 0.1, null = 0)
  expect_match(far$decision, "different")
  # A wide ROPE that contains the whole HDI: practically equivalent.
  near <- gpum_rope(fit, "mu", rope = c(2, 4))
  expect_match(near$decision, "equivalent")
})

test_that("gpum_waic prefers the better-fitting model", {
  set.seed(1)
  y <- rnorm(1000, mean = 3, sd = 2)
  # Fully normalised Gaussian log-likelihoods, so the cross-model comparison
  # is valid. The good model uses the true scale sigma = 2; the bad model
  # misreads it as sigma = 0.5. Normalising constant -0.5*log(2*pi*sigma^2):
  #   sigma = 2   -> -1.612086 ; sigma = 0.5 -> -0.225791.
  good <- gpum_model(~ -1.612086 - ((y - mu)^2) / 8, params = "mu",
                     data = "y")
  fit_g <- gpu_metropolis(good, data = list(y = y), n_iter = 3000,
                          n_chains = 8, method = "de", seed = 1,
                          backend = "cpu")
  wg <- gpum_waic(fit_g, data = list(y = y))
  expect_s3_class(wg, "gpum_waic")
  expect_true(is.finite(wg$waic))
  bad <- gpum_model(~ -0.225791 - ((y - mu)^2) / 0.5, params = "mu",
                    data = "y")
  fit_b <- gpu_metropolis(bad, data = list(y = y), n_iter = 3000,
                          n_chains = 8, method = "de", seed = 1,
                          backend = "cpu")
  wb <- gpum_waic(fit_b, data = list(y = y))
  expect_lt(wg$waic, wb$waic)
})

test_that("gpum_loo runs through the loo package", {
  skip_if_not_installed("loo")
  set.seed(1)
  y <- rnorm(500, mean = 3, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 3000, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  res <- gpum_loo(fit, data = list(y = y))
  expect_s3_class(res, "loo")
})

test_that("gpum_evidence needs a proper prior", {
  set.seed(1)
  y <- rnorm(300, mean = 3, sd = 2)
  flat <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  expect_error(gpum_evidence(flat, data = list(y = y)), "proper prior")
})

test_that("the Bayes factor favours the model whose prior matches the data", {
  set.seed(1)
  y <- rnorm(400, mean = 3, sd = 2)
  # Both models share the likelihood; they differ in where the prior sits.
  near <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y",
                     prior = ~ -((mu - 3)^2) / (2 * 25))   # N(3, 5)
  far  <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y",
                     prior = ~ -((mu + 3)^2) / (2 * 25))   # N(-3, 5)
  bf <- gpum_bayes_factor(near, far, data = list(y = y),
                          n_rungs = 8L, n_iter = 2000L, n_chains = 8L,
                          seed = 1L)
  expect_s3_class(bf, "gpum_bayes_factor")
  expect_gt(bf$bf10, 1)        # evidence favours the prior near the data
  expect_true(is.finite(bf$log_bf10))
})
