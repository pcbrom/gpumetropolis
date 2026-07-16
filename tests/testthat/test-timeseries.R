.sim_ar1 <- function(n, mu, phi, sd, seed = 3) {
  set.seed(seed)
  y <- numeric(n)
  y[1] <- mu
  for (t in 2:n) y[t] <- mu + phi * (y[t - 1] - mu) + rnorm(1, 0, sd)
  y
}

test_that("gpum_ts_model assembles the lagged conditional design", {
  y <- .sim_ar1(50, 0, 0.5, 1)
  tm <- gpum_ts_model(~ -0.5 * ((y - phi * y_lag1)^2) - 0,
                      params = "phi", series = list(y = y), order = 2)
  expect_s3_class(tm, "gpum_ts")
  expect_equal(tm$n_rows, 48L)
  expect_equal(names(tm$data), c("y", "y_lag1", "y_lag2"))
  expect_equal(tm$data$y, y[3:50])
  expect_equal(tm$data$y_lag1, y[2:49])
  expect_equal(tm$data$y_lag2, y[1:48])
  expect_output(print(tm), "Markov order: 2")
})

test_that("gpum_ts_model validates its inputs", {
  y <- rnorm(30)
  expect_error(gpum_ts_model(~ -y^2, params = "a", series = list(y),
                             order = 1), "named list")
  expect_error(gpum_ts_model(~ -y^2, params = "a", series = list(y = y),
                             order = 0), ">= 1")
  expect_error(gpum_ts_model(~ -y^2, params = "a",
                             series = list(y = y[1:2]), order = 1),
               "more than")
  expect_error(gpum_ts_model(~ -y^2, params = "a", series = list(y = y),
                             order = 1, covariates = list(x = rnorm(5))),
               "length of the series")
})

test_that("the AR(1) posterior matches the arima reference", {
  y <- .sim_ar1(600, 1.5, 0.7, 0.8)
  tm <- gpum_ts_model(
    ~ -0.5 * ((y - mu - phi * (y_lag1 - mu)) / exp(ls))^2 - ls,
    params = c("mu", "phi", "ls"), series = list(y = y), order = 1,
    prior = ~ -0.5 * (mu / 10)^2 - 0.5 * (phi / 2)^2 - 0.5 * (ls / 2)^2
  )
  fit <- gpu_metropolis(tm$model, data = tm$data, n_iter = 20000,
                        n_chains = 8, method = "mala", warmup = "auto",
                        seed = 1, backend = "cpu")
  ml <- stats::arima(y, order = c(1, 0, 0))
  expect_equal(mean(fit$draws[, , "phi"]), unname(coef(ml)["ar1"]),
               tolerance = 0.02)
  expect_equal(mean(fit$draws[, , "mu"]), unname(coef(ml)["intercept"]),
               tolerance = 0.02)
  # Posterior sd of phi against the arima standard error (Bernstein-von
  # Mises for ergodic Markov processes).
  expect_equal(sd(fit$draws[, , "phi"]),
               unname(sqrt(vcov(ml)["ar1", "ar1"])), tolerance = 0.15)
})

test_that("exogenous covariates align with the current time index", {
  y <- .sim_ar1(80, 0, 0.4, 1)
  x <- seq_len(80) / 80
  tm <- gpum_ts_model(~ -0.5 * ((y - b * x - phi * y_lag1)^2),
                      params = c("b", "phi"), series = list(y = y),
                      order = 1, covariates = list(x = x))
  expect_equal(tm$data$x, x[2:80])
})

test_that("gpum_lfo scores one-step-ahead density with an expanding window", {
  y <- .sim_ar1(120, 0, 0.6, 1)
  tm <- gpum_ts_model(
    ~ -0.5 * ((y - phi * y_lag1) / exp(ls))^2 - ls -
      0.9189385332046727,
    params = c("phi", "ls"), series = list(y = y), order = 1,
    prior = ~ -0.5 * (phi / 2)^2 - 0.5 * (ls / 2)^2
  )
  lfo <- gpum_lfo(tm, L = 110L, n_iter = 2000, n_chains = 4, seed = 1,
                  backend = "cpu")
  expect_s3_class(lfo, "gpum_lfo")
  expect_equal(lfo$n_eval, tm$n_rows - 110L)
  expect_equal(nrow(lfo$pointwise), lfo$n_eval)
  expect_true(is.finite(lfo$elpd_lfo))
  # The conditional density is normalised (the -log sqrt(2 pi) constant is
  # in the formula), so each pointwise elpd is close to the true
  # one-step-ahead log-density, bounded by the N(0, 1) entropy region.
  expect_true(all(lfo$pointwise$elpd > -6) && all(lfo$pointwise$elpd < 0))
  expect_output(print(lfo), "leave-future-out")
})

test_that("gpum_lfo validates L", {
  y <- .sim_ar1(50, 0, 0.5, 1)
  tm <- gpum_ts_model(~ -0.5 * (y - phi * y_lag1)^2, params = "phi",
                      series = list(y = y), order = 1)
  expect_error(gpum_lfo(tm, L = 49L), "L")
  expect_error(gpum_lfo(list()), "gpum_ts_model")
})
