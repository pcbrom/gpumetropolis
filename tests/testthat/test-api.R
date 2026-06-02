# Tests of the generic API: gpum_model and gpu_metropolis.

test_that("gpum_model builds and validates a model", {
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  expect_s3_class(m, "gpum_model")
  expect_equal(m$params, "mu")
  expect_equal(m$n_cols, 1L)
  # an undeclared symbol and an unsupported function are rejected
  expect_error(gpum_model(~ y + z, params = "mu", data = "y"))
  expect_error(gpum_model(~ sin(mu), params = "mu", data = "y"))
})

test_that("gpu_metropolis recovers the Gaussian-mean posterior on the CPU", {
  set.seed(1)
  y <- rnorm(5000, mean = 2.5, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(
    m, data = list(y = y),
    init = matrix(seq(-3, 8, length.out = 4), nrow = 4, ncol = 1),
    proposal_sd = 0.06, n_iter = 3000, seed = 1, backend = "cpu"
  )
  expect_s3_class(fit, "gpum_fit")
  expect_equal(dim(fit$draws), c(1500L, 4L, 1L))
  expect_equal(fit$n_iter_total, 3000L)
  expect_equal(fit$warmup, 1500L)
  expect_equal(mean(fit$draws[, , 1]), mean(y), tolerance = 0.03)
  expect_lt(rhat(fit$draws[, , 1]), 1.05)
})

test_that("the same model reproduces draws under the same seed", {
  set.seed(2)
  y <- rnorm(1000, mean = 1, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  a <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.1,
                      n_iter = 500, n_chains = 2, seed = 7, backend = "cpu")
  b <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.1,
                      n_iter = 500, n_chains = 2, seed = 7, backend = "cpu")
  expect_identical(a$draws, b$draws)
})

test_that("gpu_metropolis rejects malformed input", {
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  expect_error(gpu_metropolis(m, data = list(z = 1:10)))
  expect_error(gpu_metropolis(m, data = list(y = rnorm(10)),
                              proposal_sd = -1))
})

test_that("gpu_metropolis accepts a per-chain proposal_sd matrix", {
  set.seed(7)
  y <- rnorm(1000, mean = 0, sd = 1)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  n_chains <- 4L
  per_chain_sd <- matrix(c(0.05, 0.10, 0.20, 0.40),
                         nrow = n_chains, ncol = 1L)
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = per_chain_sd,
                        n_iter = 2000, n_chains = n_chains, warmup = 0,
                        seed = 1, backend = "cpu")
  expect_equal(dim(fit$draws), c(2000L, n_chains, 1L))
  # Each chain uses a different proposal scale, so the accept rates must
  # span a wide range: tight proposals accept often, wide ones rarely.
  expect_gt(fit$accept_rate[1] - fit$accept_rate[4], 0.4)
})

test_that("a per-chain proposal_sd with a row of zeros is rejected", {
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  bad <- matrix(c(0.1, 0.0, 0.2, 0.1), nrow = 4L, ncol = 1L)
  expect_error(
    gpu_metropolis(m, data = list(y = rnorm(20)), proposal_sd = bad,
                   n_chains = 4L, backend = "cpu"),
    "positive and finite"
  )
})

test_that("a per-chain proposal_sd of the wrong shape is rejected", {
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  bad <- matrix(0.1, nrow = 3L, ncol = 1L)
  expect_error(
    gpu_metropolis(m, data = list(y = rnorm(20)), proposal_sd = bad,
                   n_chains = 4L, backend = "cpu"),
    "n_chains.*by.*n_params"
  )
})

test_that("the CUDA backend matches the CPU backend distributionally", {
  testthat::skip_on_cran()
  cuda_ok <- tryCatch({
    mm <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
    gpu_metropolis(mm, data = list(y = rnorm(100)), n_iter = 10,
                   n_chains = 2, backend = "cuda")
    TRUE
  }, error = function(e) FALSE)
  testthat::skip_if(!cuda_ok, "no CUDA backend available")

  set.seed(3)
  y <- rnorm(4000, mean = 4, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  cpu <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.06,
                        n_iter = 3000, n_chains = 4, seed = 5,
                        backend = "cpu")
  cuda <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.06,
                         n_iter = 3000, n_chains = 4, seed = 5,
                         backend = "cuda")
  res <- ks_equivalence(cpu$draws[, , 1], cuda$draws[, , 1])
  expect_true(res$equivalent)
})

test_that("the Vulkan backend matches the CPU backend distributionally", {
  testthat::skip_on_cran()
  vulkan_ok <- tryCatch({
    mm <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
    gpu_metropolis(mm, data = list(y = rnorm(100)), n_iter = 10,
                   n_chains = 2, backend = "vulkan")
    TRUE
  }, error = function(e) FALSE)
  testthat::skip_if(!vulkan_ok, "no Vulkan backend available")

  set.seed(4)
  y <- rnorm(4000, mean = 4, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  cpu <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.06,
                        n_iter = 3000, n_chains = 4, seed = 5,
                        backend = "cpu")
  vulkan <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.06,
                           n_iter = 3000, n_chains = 4, seed = 5,
                           backend = "vulkan")
  res <- ks_equivalence(cpu$draws[, , 1], vulkan$draws[, , 1])
  expect_true(res$equivalent)
})
