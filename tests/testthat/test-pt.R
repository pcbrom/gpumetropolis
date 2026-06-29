# Parallel tempering integration tests. The target is the bimodal
# y ~ Normal(|mu|, 1) likelihood (M2 of the benchmark); without swaps a
# random-walk chain stays in the mode it started in. With PT the cold
# chain visits both modes through swaps from hotter chains.

test_that(".pt_default_ladder produces a monotone geometric ladder", {
  T <- gpumetropolis:::.pt_default_ladder(8L, t_max = 10)
  expect_length(T, 8L)
  expect_equal(T[1L], 1)
  expect_equal(T[8L], 10)
  expect_true(all(diff(T) > 0))
  ratios <- T[-1L] / T[-length(T)]
  expect_lt(max(abs(ratios - ratios[1L])), 1e-9)
})

test_that(".pt_swap_draws is deterministic from the seed and does not pollute the global RNG", {
  set.seed(42)
  before <- stats::runif(1)
  u1 <- gpumetropolis:::.pt_swap_draws(seed = 1, batch = 1, n_draws = 5)
  u2 <- gpumetropolis:::.pt_swap_draws(seed = 1, batch = 1, n_draws = 5)
  expect_identical(u1, u2)
  u3 <- gpumetropolis:::.pt_swap_draws(seed = 1, batch = 2, n_draws = 5)
  expect_false(identical(u1, u3))
  set.seed(42)
  before2 <- stats::runif(1)
  expect_identical(before, before2)
})

test_that("PT cold chain covers both modes of the M2 bimodal target", {
  set.seed(1)
  y <- rnorm(400, mean = 3, sd = 1)
  m <- gpum_model(
    loglik = ~ log(exp(-((y - mu)^2) / 2) + exp(-((y + mu)^2) / 2)),
    params = "mu", data = "y"
  )
  init <- matrix(seq(-6, 6, length.out = 8), nrow = 8, ncol = 1)
  fit <- gpu_metropolis(m, data = list(y = y), init = init,
                        proposal_sd = 0.15, n_iter = 4000,
                        method = "pt", seed = 1, backend = "cpu")
  expect_identical(fit$method, "pt")
  expect_equal(length(fit$temperatures), 8L)
  expect_equal(fit$temperatures[1L], 1)
  cold <- as.vector(fit$draws[, 1L, 1L])
  expect_true(any(cold > 1.5))
  expect_true(any(cold < -1.5))
  expect_equal(mean(cold[cold > 0]), 3, tolerance = 0.4)
  expect_equal(mean(cold[cold < 0]), -3, tolerance = 0.4)
})

test_that("PT respects a user-supplied temperature ladder and swap_every", {
  set.seed(2)
  y <- rnorm(200, mean = 2, sd = 1)
  m <- gpum_model(loglik = ~ -((y - mu)^2) / 2, params = "mu", data = "y")
  T <- c(1, 2, 4, 8)
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 500, n_chains = 4,
                        method = "pt", temperatures = T, swap_every = 5,
                        adapt = FALSE, seed = 3, backend = "cpu")
  expect_equal(fit$temperatures, T)
  expect_equal(fit$swap_every, 5L)
  expect_identical(dim(fit$adaptation$swap_history),
                   c(length(T) - 1L, fit$adaptation$n_batches))
  expect_true(all(diff(fit$adaptation$batch_sizes[-fit$adaptation$n_batches]) ==
                  0))
})

test_that("PT rejects malformed temperature input", {
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  expect_error(
    gpu_metropolis(m, data = list(y = rnorm(50)), n_iter = 100, n_chains = 4,
                   method = "pt", temperatures = c(1, 2), backend = "cpu"),
    "one value per chain"
  )
  expect_error(
    gpu_metropolis(m, data = list(y = rnorm(50)), n_iter = 100, n_chains = 4,
                   method = "pt", temperatures = c(1, 2, -1, 3),
                   backend = "cpu"),
    "positive and finite"
  )
})

test_that("PT fit prints with the method label and gpum_diagnose collapses to the cold chain", {
  set.seed(4)
  y <- rnorm(150, mean = 0, sd = 1)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 800, n_chains = 4,
                        method = "pt", seed = 5, backend = "cpu")
  out <- capture.output(print(fit))
  expect_true(any(grepl("method      : pt", out)))
  expect_true(any(grepl("swap accept", out)))

  diag_out <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_equal(nrow(diag_out$summary), 1L)
  expect_named(diag_out, c("summary", "verdict", "adaptation",
                           "adaptation_hint", "swap_hint", "de_hint"))
})
