# Simulate from a family without depending on the `copula` package: the
# conditional-inverse sampler in the package itself generates pseudo-obs at
# a known parameter, and the analytic Kendall tau gives the external anchor.
.rcop <- function(family, theta, n, seed = 1) {
  set.seed(seed)
  spec <- gpumetropolis:::.copula_family(family)
  z <- switch(family,
    gumbel = log(theta - 1), clayton = log(theta),
    frank = log(theta), gaussian = theta / sqrt(1 - theta^2))
  u <- stats::runif(n); w <- stats::runif(n)
  v <- gpumetropolis:::.copula_cond_inv(family, u, w, theta)
  data.frame(x = stats::qnorm(u), y = stats::qexp(v))
}

test_that("all four copula families compile within the DSL stack limit", {
  for (fam in c("gumbel", "clayton", "frank", "gaussian")) {
    sp <- gpumetropolis:::.copula_family(fam)
    m <- gpum_model(sp$loglik, params = "z", data = sp$data)
    expect_lte(m$loglik$depth, 32L)
  }
})

test_that("the copula log-densities integrate to one on the unit square", {
  # A valid copula density integrates to 1 over [0,1]^2. Check on a grid.
  for (fam in c("gumbel", "clayton", "frank", "gaussian")) {
    sp <- gpumetropolis:::.copula_family(fam)
    m <- gpum_model(sp$loglik, params = "z", data = sp$data)
    theta <- switch(fam, gumbel = 2, clayton = 2, frank = 4, gaussian = 0.5)
    z <- switch(fam, gumbel = log(theta - 1), clayton = log(theta),
                frank = log(theta), gaussian = theta / sqrt(1 - theta^2))
    g <- seq(0.005, 0.995, length.out = 120)
    gg <- expand.grid(u = g, v = g)
    cd <- sp$prep(gg$u, gg$v)
    d <- gpumetropolis:::.gpum_data_flat(m, cd)
    ll <- vapply(seq_len(nrow(gg)), function(i) {
      di <- lapply(cd, `[`, i)
      dd <- gpumetropolis:::.gpum_data_flat(m, di)
      gpumetropolis:::rust_loglik_batch(m$loglik$code, m$loglik$consts, 1L,
        as.numeric(dd$flat), m$n_cols, dd$n_obs, z)
    }, numeric(1))
    integral <- mean(exp(ll))  # cell area is (1/120)^2 * 120^2 = 1
    expect_equal(integral, 1, tolerance = 0.05)
  }
})

test_that("gpum_copula validates its input", {
  expect_error(gpum_copula(matrix(1, 10, 3)), "exactly two columns")
  expect_error(gpum_copula(matrix(1, 3, 2)), "at least five rows")
})

test_that("gpum_copula recovers the parameter and tau of each family", {
  cfg <- list(gumbel = 2.0, clayton = 2.0, frank = 5.0, gaussian = 0.6)
  tau_fun <- list(
    gumbel = function(th) (th - 1) / th,
    clayton = function(th) th / (th + 2),
    frank = gpumetropolis:::.frank_tau,
    gaussian = function(rho) (2 / pi) * asin(rho))
  for (fam in names(cfg)) {
    th <- cfg[[fam]]
    dat <- .rcop(fam, th, 800)
    fit <- gpum_copula(dat, family = fam, n_iter = 6000, n_chains = 4,
                       seed = 1, backend = "cpu")
    expect_s3_class(fit, "gpum_copula")
    expect_lt(fit$rhat_max, 1.05)
    # The model-implied tau posterior brackets the empirical tau.
    expect_gt(fit$tau_empirical, fit$tau["q2.5"])
    expect_lt(fit$tau_empirical, fit$tau["q97.5"])
    # The parameter posterior brackets the analytic tau's parameter.
    expect_gt(fit$tau["q97.5"], tau_fun[[fam]](th) - 0.08)
    expect_lt(fit$tau["q2.5"], tau_fun[[fam]](th) + 0.08)
  }
})

test_that("family auto-selection recovers the generating family", {
  # Gumbel data (upper-tail dependence) should be selected as Gumbel.
  dat <- .rcop("gumbel", 2.5, 800)
  fit <- gpum_copula(dat, family = "auto", n_iter = 5000, n_chains = 4,
                     seed = 1, backend = "cpu")
  expect_equal(fit$family, "gumbel")
  expect_true(is.data.frame(fit$comparison))
  expect_equal(nrow(fit$comparison), 4L)
  expect_equal(fit$comparison$family[1], "gumbel")
  expect_true(all(fit$comparison$delta >= 0))
})

test_that("tail dependence is the qualitative signature of the family", {
  dat_g <- .rcop("gumbel", 3, 600)
  fg <- gpum_copula(dat_g, family = "gumbel", n_iter = 4000, n_chains = 4,
                    seed = 1, backend = "cpu")
  expect_gt(fg$tail_upper["mean"], 0.3)   # Gumbel: upper-tail dependence
  expect_equal(unname(fg$tail_lower["mean"]), 0)
  dat_c <- .rcop("clayton", 3, 600)
  fc <- gpum_copula(dat_c, family = "clayton", n_iter = 4000, n_chains = 4,
                    seed = 1, backend = "cpu")
  expect_gt(fc$tail_lower["mean"], 0.3)   # Clayton: lower-tail dependence
  expect_equal(unname(fc$tail_upper["mean"]), 0)
})

test_that("kendall_tau and tail_dependence accessors work", {
  dat <- .rcop("gaussian", 0.5, 500)
  fit <- gpum_copula(dat, family = "gaussian", n_iter = 4000, n_chains = 4,
                     seed = 1, backend = "cpu")
  kt <- kendall_tau(fit)
  expect_named(kt, c("mean", "q2.5", "median", "q97.5"))
  td <- tail_dependence(fit)
  expect_named(td, c("lower", "upper"))
  expect_error(kendall_tau(list()), "gpum_copula")
})

test_that("the object carries a text diagnostic and a visual method", {
  dat <- .rcop("gumbel", 2, 500)
  fit <- gpum_copula(dat, family = "gumbel", n_iter = 4000, n_chains = 4,
                     seed = 1, backend = "cpu")
  out <- capture.output(print(fit))
  expect_true(any(grepl("diagnostic", out)))
  expect_true(any(grepl("empirical", out)))
  # The plot method runs without error to a null device.
  grDevices::pdf(NULL)
  expect_invisible(plot(fit))
  grDevices::dev.off()
})
