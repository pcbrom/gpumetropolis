# Tests for gpum_diagnose: structure of the returned data, the verdict
# logic, and that the plot path does not error on the canonical fit.

test_that("gpum_diagnose returns the summary and the verdict", {
  set.seed(1)
  y <- rnorm(800, mean = 1, sd = 1)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.1,
                        n_iter = 2000, n_chains = 4, warmup = 1000,
                        seed = 1, backend = "cpu")
  out <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_named(out, c("summary", "verdict", "adaptation",
                      "adaptation_hint", "swap_hint", "de_hint"))
  expect_equal(nrow(out$summary), 1L)
  expect_true(all(c("mean", "sd", "q2.5", "q50", "q97.5", "Rhat", "ESS",
                    "MCSE") %in% names(out$summary)))
  expect_true(out$verdict[1L] %in%
                c("Converged", "Inconclusive", "Not converged"))
})

test_that("gpum_diagnose reports Converged on a clean Gaussian-mean fit", {
  set.seed(2)
  y <- rnorm(2000, mean = 5, sd = 1.5)
  m <- gpum_model(~ -((y - mu)^2) / (2 * 1.5^2), params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.05,
                        n_iter = 6000, n_chains = 4, warmup = 3000,
                        seed = 11, backend = "cpu")
  out <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_equal(out$verdict[1L], "Converged")
  expect_lt(out$summary$Rhat[1L], 1.01)
  expect_gt(out$summary$ESS[1L], 400)
})

test_that("gpum_diagnose flags Not converged on chains pinned far apart", {
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = c(0)),
                        init = matrix(c(-50, 50), nrow = 2, ncol = 1),
                        proposal_sd = 1e-6,
                        n_iter = 800, n_chains = 2, warmup = 0,
                        adapt = FALSE, seed = 1, backend = "cpu")
  out <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_equal(out$verdict[1L], "Not converged")
})

test_that("gpum_diagnose handles a two-parameter fit", {
  set.seed(3)
  y <- rnorm(500, mean = 2, sd = 1.2)
  m <- gpum_model(
    loglik = ~ -((y - mu)^2) / (2 * exp(2 * ls)) - ls,
    params = c("mu", "ls"), data = "y"
  )
  fit <- gpu_metropolis(m, data = list(y = y),
                        proposal_sd = c(0.05, 0.05),
                        n_iter = 2000, n_chains = 4, warmup = 1000,
                        seed = 5, backend = "cpu")
  out <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_equal(out$summary$parameter, c("mu", "ls"))
  expect_equal(nrow(out$summary), 2L)
})

test_that("gpum_diagnose emits an adaptation_hint when the warmup is too short", {
  set.seed(8)
  y <- rnorm(2000, mean = 3, sd = 0.05)   # very tight posterior
  m <- gpum_model(~ -((y - mu)^2) / (2 * 0.05^2), params = "mu", data = "y")
  # Start with a wide proposal_sd and a short warmup: adaptation cannot
  # close the gap before the warmup boundary, so the hint must fire.
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 5,
                        n_iter = 600, n_chains = 4, warmup = 300,
                        seed = 1, backend = "cpu")
  out <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_false(is.null(out$adaptation_hint))
  expect_match(out$adaptation_hint, "still climbing")
  expect_match(out$adaptation_hint, sprintf("warmup = %d", 2L * fit$warmup))
})

test_that("gpum_diagnose stays silent on a well-tuned warmup", {
  set.seed(9)
  y <- rnorm(800, mean = 0, sd = 1)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.05,
                        n_iter = 8000, n_chains = 4, warmup = 4000,
                        seed = 1, backend = "cpu")
  out <- gpum_diagnose(fit, plot = FALSE, return_data = TRUE)
  expect_null(out$adaptation_hint)
})

test_that("gpum_diagnose plot path does not error", {
  set.seed(4)
  y <- rnorm(200, mean = 0, sd = 1)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.1,
                        n_iter = 600, n_chains = 4, warmup = 300,
                        seed = 1, backend = "cpu")
  grDevices::pdf(tempfile(fileext = ".pdf"))
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_error(
    utils::capture.output(gpum_diagnose(fit, plot = TRUE)),
    NA
  )
})
