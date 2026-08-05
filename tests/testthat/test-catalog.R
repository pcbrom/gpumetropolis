test_that("the lgamma opcode matches R and its gradient matches digamma", {
  m <- gpum_model(~ a * log(b) - lgamma(a) + (a - 1) * log(y) - b * y,
                  params = c("a", "b"), data = "y")
  yv <- c(0.5, 1.2, 2.0, 3.5)
  pt <- c(2.5, 1.3)
  val <- rust_loglik_batch(m$loglik$code, m$loglik$consts, 2L, yv, 1L,
                           length(yv), pt)
  expect_equal(val, sum(dgamma(yv, shape = pt[1], rate = pt[2], log = TRUE)),
               tolerance = 1e-10)
  g <- rust_grad_batch(m$loglik$code, m$loglik$consts, 2L, yv, 1L,
                       length(yv), pt)
  # analytic gradient: d/da = n log b - n digamma(a) + sum log y; d/db = n a/b - sum y
  ga <- length(yv) * log(pt[2]) - length(yv) * digamma(pt[1]) + sum(log(yv))
  gb <- length(yv) * pt[1] / pt[2] - sum(yv)
  expect_equal(g, c(ga, gb), tolerance = 1e-6)
})

test_that("every catalogue family compiles within the DSL stack limit", {
  reg <- gpumetropolis:::.catalog_registry()
  for (nm in names(reg)) {
    sp <- reg[[nm]]
    pri <- if (is.function(sp$prior)) {
      stats::as.formula(paste("~", deparse(sp$prior(rnorm(50)),
                                           width.cutoff = 500)))
    } else sp$prior
    m <- gpum_model(stats::as.formula(paste("~", deparse(sp$loglik,
                                                        width.cutoff = 500))),
                    params = sp$params, data = "y", prior = pri)
    expect_lte(m$loglik$depth, 32L)
    expect_equal(m$n_params, length(sp$params))
  }
})

test_that("support detection routes to the right family bucket", {
  expect_true("beta" %in% names(gpumetropolis:::.catalog_registry()))
  expect_equal(gpumetropolis:::.catalog_support(runif(50) * 0.5)[1], "unit")
  expect_equal(gpumetropolis:::.catalog_support(rexp(50))[1], "positive")
  expect_equal(gpumetropolis:::.catalog_support(rnorm(50)), "real")
})

test_that("modality detection separates unimodal from bimodal", {
  set.seed(1)
  expect_false(gpumetropolis:::.catalog_multimodal(rgamma(400, 3, 2))$multimodal)
  expect_false(gpumetropolis:::.catalog_multimodal(rnorm(400))$multimodal)
  expect_true(gpumetropolis:::.catalog_multimodal(
    c(rnorm(200, -3, 0.6), rnorm(200, 3, 0.6)))$multimodal)
})

test_that("the catalogue recovers the generating family", {
  set.seed(1)
  cases <- list(
    gamma = rgamma(400, shape = 3, rate = 2),
    normal = rnorm(400, 5, 2),
    lognormal = rlnorm(400, 1, 0.5),
    beta = rbeta(400, 2, 5)
  )
  for (nm in names(cases)) {
    r <- gpum_fit_catalog(cases[[nm]], n_iter = 5000, n_chains = 4, seed = 1,
                          backend = "cpu")
    expect_s3_class(r, "gpum_catalog")
    expect_equal(r$best, nm)
    expect_false(r$modality$multimodal)
  }
})

test_that("heavy-tailed data selects the Student-t (exercising lgamma)", {
  set.seed(1)
  yt <- 2 + 1.5 * rt(600, df = 4)
  r <- gpum_fit_catalog(yt, catalog = c("normal", "student_t", "logistic"),
                        n_iter = 6000, n_chains = 4, seed = 1, backend = "cpu")
  expect_equal(r$best, "student_t")
})

test_that("bimodal data selects the two-component mixture", {
  set.seed(1)
  y <- c(rnorm(250, -2, 0.6), rnorm(250, 3, 0.8))
  r <- gpum_fit_catalog(y, n_iter = 6000, n_chains = 6, seed = 1,
                        backend = "cpu")
  expect_equal(r$best, "gmm2")
  expect_true(r$modality$multimodal)
  # Relabeling: component 1 is the lower-location one after post-processing.
  expect_lt(r$point$mu1, r$point$mu2)
})

test_that("marginal_cdf matches the true CDF and validates input", {
  set.seed(1)
  y <- rgamma(400, shape = 3, rate = 2)
  r <- gpum_fit_catalog(y, catalog = "gamma", n_iter = 5000, n_chains = 4,
                        seed = 1, backend = "cpu")
  fx <- marginal_cdf(r, c(0.5, 1, 2))
  expect_equal(fx, pgamma(c(0.5, 1, 2), 3, 2), tolerance = 0.03)
  expect_true(is.function(marginal_cdf(r)))
  expect_error(marginal_cdf(list()), "gpum_fit_catalog")
})

test_that("the object carries a text diagnostic and a visual method", {
  set.seed(1)
  y <- rnorm(300, 0, 1)
  r <- gpum_fit_catalog(y, catalog = c("normal", "logistic"), n_iter = 4000,
                        n_chains = 4, seed = 1, backend = "cpu")
  out <- capture.output(print(r))
  expect_true(any(grepl("diagnostic", out)))
  expect_true(any(grepl("KS distance", out)))
  expect_true(any(grepl("ranking", out)))
  grDevices::pdf(NULL)
  expect_invisible(plot(r))
  grDevices::dev.off()
})

test_that("fitted marginals feed the copula transform end to end", {
  skip_if_not_installed("copula")
  set.seed(2)
  n <- 500
  u <- runif(n); w <- runif(n)
  v <- gpumetropolis:::.copula_cond_inv("clayton", u, w, 2)
  x <- qgamma(u, 3, 2); yy <- qlnorm(v, 0, 0.5)
  mx <- gpum_fit_catalog(x, n_iter = 4000, n_chains = 4, seed = 1,
                         backend = "cpu")
  my <- gpum_fit_catalog(yy, n_iter = 4000, n_chains = 4, seed = 1,
                         backend = "cpu")
  ux <- marginal_cdf(mx, x); uy <- marginal_cdf(my, yy)
  clamp <- function(p) pmin(pmax(p, 1e-4), 1 - 1e-4)
  cop <- gpum_copula(data.frame(a = qnorm(clamp(ux)), b = qnorm(clamp(uy))),
                     family = "clayton", n_iter = 5000, n_chains = 4,
                     seed = 1, backend = "cpu")
  # The dependence is recovered through the fitted-marginal transform.
  expect_gt(cop$tau["mean"], 0.35)
  expect_lt(cop$tau["mean"], 0.65)
})

test_that("a GPU backend failure is reported with an actionable hint", {
  # The CPU path never triggers the guard; a requested-but-absent backend
  # is caught before launch with the availability message.
  m <- gpum_model(~ -0.5 * (mu)^2, params = "mu")
  avail <- gpumetropolis:::rust_available_backends()
  if (!("vulkan" %in% avail)) {
    expect_error(
      gpu_metropolis(m, n_iter = 100, n_chains = 2, warmup = 50,
                     backend = "vulkan"),
      "not available"
    )
  }
  # The translator itself maps a known GPU error to a hint.
  g <- gpumetropolis:::.gpum_backend_guard
  err <- tryCatch(
    g(stop("nvrtc: error: invalid value for --gpu-architecture"),
      backend = "cuda"),
    error = function(e) conditionMessage(e))
  expect_match(err, "CUDA 12.8|toolkit")
})
