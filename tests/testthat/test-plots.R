# Tests for the plotting layer. Plots are sent to a throwaway device, so the
# checks are that the calls run without error and that the data-returning
# helpers produce the expected structure.

make_fit2 <- function() {
  set.seed(1)
  rho <- 0.8
  Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
  n <- 400
  y <- matrix(rnorm(n * 2), n, 2) %*% chol(Sigma)
  y[, 1] <- y[, 1] + 2
  y[, 2] <- y[, 2] - 1
  Si <- solve(Sigma)
  loglik <- stats::as.formula(sprintf(
    "~ -0.5 * (%.6f*(y1-mu1)^2 + 2*%.6f*(y1-mu1)*(y2-mu2) + %.6f*(y2-mu2)^2)",
    Si[1, 1], Si[1, 2], Si[2, 2]))
  m <- gpum_model(loglik, params = c("mu1", "mu2"), data = c("y1", "y2"))
  gpu_metropolis(m, data = list(y1 = y[, 1], y2 = y[, 2]), n_iter = 4000,
                 n_chains = 16, method = "de", seed = 1, backend = "cpu")
}

test_that("gpum_region returns credible-region contours", {
  fit <- make_fit2()
  reg <- gpum_region(fit, c("mu1", "mu2"), level = 0.95)
  expect_s3_class(reg, "gpum_region")
  expect_gt(length(reg$contours), 0L)
  expect_true(all(c("x", "y") %in% names(reg$contours[[1L]])))
})

test_that("the plotting functions run without error", {
  fit <- make_fit2()
  pdf(tempfile(fileext = ".pdf"))
  on.exit(dev.off(), add = TRUE)
  expect_no_error(gpum_pairs(fit))
  expect_no_error(gpum_surface(fit, c("mu1", "mu2"), type = "both"))
  expect_no_error(gpum_surface(fit, c("mu1", "mu2"), type = "contour"))
  expect_no_error(gpum_surface(fit, c("mu1", "mu2"), type = "persp"))
  reg <- gpum_region(fit, c("mu1", "mu2"))
  plot(reg)
  expect_no_error(lines(reg, col = "red"))
})

test_that("gpum_pairs overlays the Cramer-Rao ellipse when supplied", {
  set.seed(1)
  rho <- 0.8
  Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
  n <- 400
  y <- matrix(rnorm(n * 2), n, 2) %*% chol(Sigma)
  y[, 1] <- y[, 1] + 2
  y[, 2] <- y[, 2] - 1
  Si <- solve(Sigma)
  loglik <- stats::as.formula(sprintf(
    "~ -0.5 * (%.6f*(y1-mu1)^2 + 2*%.6f*(y1-mu1)*(y2-mu2) + %.6f*(y2-mu2)^2)",
    Si[1, 1], Si[1, 2], Si[2, 2]))
  m <- gpum_model(loglik, params = c("mu1", "mu2"), data = c("y1", "y2"))
  dat <- list(y1 = y[, 1], y2 = y[, 2])
  fit <- gpu_metropolis(m, data = dat, n_iter = 4000, n_chains = 16,
                        method = "de", seed = 1, backend = "cpu")
  cr <- gpum_crlb(fit, data = dat)
  expect_true(cr$applicable)
  pdf(tempfile(fileext = ".pdf"))
  on.exit(dev.off(), add = TRUE)
  expect_no_error(gpum_pairs(fit, crlb = cr))
  expect_no_error(plot(cr))
})

test_that("plot methods for the hypothesis tools run", {
  fit <- make_fit2()
  pdf(tempfile(fileext = ".pdf"))
  on.exit(dev.off(), add = TRUE)
  expect_no_error(plot(gpum_hypothesis(fit, "mu1", lower = 0)))
  expect_no_error(plot(gpum_rope(fit, "mu1", rope = 0.1)))
})

test_that("gpum_ppc draws a predictive sample and gpum_density_compare plots it", {
  set.seed(1)
  y <- rnorm(500, mean = 3, sd = 2)
  m <- gpum_model(~ -((y - mu)^2) / 8, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), n_iter = 2000, n_chains = 8,
                        method = "de", seed = 1, backend = "cpu")
  pp <- gpum_ppc(fit, function(p) stats::rnorm(1, p["mu"], 2), n = 1000L)
  expect_length(pp, 1000L)
  expect_true(all(is.finite(pp)))
  # the predictive mean tracks the data mean
  expect_lt(abs(mean(pp) - mean(y)), 0.3)
  pdf(tempfile(fileext = ".pdf"))
  on.exit(dev.off(), add = TRUE)
  expect_no_error(gpum_density_compare(y, list(model = pp)))
  expect_no_error(gpum_density_compare(y, pp))
})
