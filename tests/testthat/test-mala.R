test_that("the reverse-mode gradient matches finite differences", {
  y <- c(3.2, 4.1, 3.8, 4.5, 3.6, 4.0, 3.9)
  gum <- gpum_model(~ -(y - mu) / exp(lb) - exp(-(y - mu) / exp(lb)) - lb,
                    params = c("mu", "lb"), data = "y")
  ll <- gum$loglik
  pt <- c(3.9, -0.7)
  g_ad <- rust_grad_batch(ll$code, ll$consts, 2L, y, 1L, length(y), pt)
  h <- 1e-6
  fd <- vapply(1:2, function(j) {
    e <- numeric(2); e[j] <- h
    (rust_loglik_batch(ll$code, ll$consts, 2L, y, 1L, length(y), pt + e) -
       rust_loglik_batch(ll$code, ll$consts, 2L, y, 1L, length(y),
                         pt - e)) / (2 * h)
  }, numeric(1))
  expect_equal(g_ad, fd, tolerance = 1e-6)
})

test_that("the gradient covers every opcode (pow, sqrt, div, neg)", {
  y <- c(3.2, 4.1, 3.8)
  m2 <- gpum_model(~ -sqrt((y - a)^2 + 1) - log(1 + (y - a)^2 / b^2),
                   params = c("a", "b"), data = "y")
  ll <- m2$loglik
  p2 <- c(3.5, 1.3)
  g_ad <- rust_grad_batch(ll$code, ll$consts, 2L, y, 1L, length(y), p2)
  h <- 1e-6
  fd <- vapply(1:2, function(j) {
    e <- numeric(2); e[j] <- h
    (rust_loglik_batch(ll$code, ll$consts, 2L, y, 1L, length(y), p2 + e) -
       rust_loglik_batch(ll$code, ll$consts, 2L, y, 1L, length(y),
                         p2 - e)) / (2 * h)
  }, numeric(1))
  expect_equal(g_ad, fd, tolerance = 1e-6)
})

test_that("mala recovers a known Gaussian posterior at its target rate", {
  set.seed(1)
  y <- rnorm(400, 2.5, 1.0)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  f <- gpu_metropolis(m, data = list(y = y), n_iter = 8000, n_chains = 4,
                      method = "mala", seed = 1, backend = "cpu")
  expect_equal(f$method, "mala")
  expect_equal(mean(f$draws), mean(y), tolerance = 0.01)
  expect_equal(sd(as.vector(f$draws)), 1 / sqrt(400), tolerance = 0.15)
  # Acceptance sits near the MALA optimum, not the random-walk one.
  expect_true(all(f$accept_rate > 0.45 & f$accept_rate < 0.70))
})

test_that("mala beats rwm per draw on a correlated non-conjugate target", {
  skip_if_not_installed("evd")
  skip_if_not_installed("coda")
  y <- as.numeric(evd::portpirie)
  gum <- gpum_model(~ -(y - mu) / exp(lb) - exp(-(y - mu) / exp(lb)) - lb,
                    params = c("mu", "lb"), data = "y")
  ess_pd <- function(f) {
    dr <- as.vector(f$draws[, , "mu"])
    as.numeric(coda::effectiveSize(dr)) / length(dr)
  }
  fm <- gpu_metropolis(gum, data = list(y = y), n_iter = 10000, n_chains = 4,
                       proposal_sd = c(0.04, 0.1), method = "mala",
                       seed = 1, backend = "cpu")
  fr <- gpu_metropolis(gum, data = list(y = y), n_iter = 10000, n_chains = 4,
                       proposal_sd = c(0.04, 0.1), method = "rwm",
                       seed = 1, backend = "cpu")
  expect_equal(mean(fm$draws[, , "mu"]), mean(fr$draws[, , "mu"]),
               tolerance = 0.01)
  expect_gt(ess_pd(fm), 2 * ess_pd(fr))
})

test_that("mala refuses a GPU backend in this release", {
  m <- gpum_model(~ -(mu^2) / 2, params = "mu")
  expect_error(
    gpu_metropolis(m, n_iter = 100, n_chains = 2, method = "mala",
                   backend = "cuda"),
    "cpu|not available"
  )
})
