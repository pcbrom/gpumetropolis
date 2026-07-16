# Tests for the adaptive warmup primitives. These primitives are R-only
# in this development cut; the Rust kernel still consumes a diagonal
# proposal, with the full Cholesky path opening once the kernel accepts
# a per-chain L matrix.

test_that(".am_welford_update reproduces cov() in d = 1", {
  set.seed(1)
  x <- rnorm(500)
  state <- .am_init_state(1L)
  for (v in x) state <- .am_welford_update(state, v)
  expect_equal(state$mean, mean(x))
  expect_equal(as.numeric(.am_covariance(state)), var(x))
})

test_that(".am_welford_update_batch reproduces cov() in d = 3", {
  set.seed(2)
  X <- matrix(rnorm(900), ncol = 3)
  state <- .am_init_state(3L)
  state <- .am_welford_update_batch(state, X)
  expect_equal(state$mean, colMeans(X))
  expect_equal(.am_covariance(state), cov(X))
})

test_that(".am_welford_update is order-invariant within numerical noise", {
  set.seed(3)
  X <- matrix(rnorm(600), ncol = 2)
  s1 <- .am_welford_update_batch(.am_init_state(2L), X)
  s2 <- .am_welford_update_batch(.am_init_state(2L), X[sample(nrow(X)), ])
  expect_equal(s1$mean, s2$mean, tolerance = 1e-12)
  expect_equal(.am_covariance(s1), .am_covariance(s2), tolerance = 1e-12)
})

test_that(".am_covariance returns the seed Sigma before two samples land", {
  state <- .am_init_state(2L, sigma_init = 0.5)
  expect_equal(.am_covariance(state), diag(0.5, 2))
  state <- .am_welford_update(state, c(1, 2))
  expect_equal(.am_covariance(state), diag(0.5, 2))
})

test_that(".am_robbins_monro_scale grows the scale when accept exceeds the target", {
  for (d in c(1L, 2L, 5L)) {
    target <- .am_target_accept(d)
    grown <- .am_robbins_monro_scale(1.0, accept_rate = target + 0.1,
                                     batch_idx = 1L, d = d)
    shrunk <- .am_robbins_monro_scale(1.0, accept_rate = target - 0.1,
                                      batch_idx = 1L, d = d)
    expect_gt(grown, 1.0)
    expect_lt(shrunk, 1.0)
  }
})

test_that(".am_robbins_monro_scale step size vanishes as the batch index grows", {
  # gamma_t = t^{-2/3}: bigger batch index gives smaller move per unit gap.
  early <- .am_robbins_monro_scale(1.0, accept_rate = 0.5, batch_idx = 1L,
                                   d = 2L) - 1.0
  late <- .am_robbins_monro_scale(1.0, accept_rate = 0.5, batch_idx = 100L,
                                  d = 2L) - 1.0
  expect_gt(early, late)
  expect_gt(late, 0)
})

test_that(".am_cholesky reconstructs Sigma up to the regulariser", {
  set.seed(4)
  A <- matrix(rnorm(9), 3, 3)
  Sigma <- crossprod(A) + diag(0.1, 3)
  L <- .am_cholesky(Sigma, eps = 1e-12)
  expect_equal(L %*% t(L), Sigma, tolerance = 1e-6)
  expect_true(all(L[upper.tri(L)] == 0))
})

test_that(".am_cholesky regularises a singular matrix without erroring", {
  Sigma <- matrix(c(1, 1, 1, 1), nrow = 2)
  L <- .am_cholesky(Sigma, eps = 1e-3)
  expect_equal(dim(L), c(2L, 2L))
  expect_true(all(is.finite(L)))
  recon <- L %*% t(L)
  expect_true(all(eigen(recon, symmetric = TRUE, only.values = TRUE)$values > 0))
})

test_that("adaptive warmup rescues a deliberately bad proposal_sd", {
  set.seed(42)
  y <- rnorm(500, mean = 5, sd = 1)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  # An absurdly tight proposal_sd would freeze a non-adaptive chain.
  # Adaptation must inflate the scale during warmup and recover the mode.
  fit_adapt <- gpu_metropolis(m, data = list(y = y), proposal_sd = 1e-4,
                              n_iter = 2000, n_chains = 4, warmup = 1000,
                              adapt = TRUE, seed = 1, backend = "cpu")
  fit_off <- gpu_metropolis(m, data = list(y = y), proposal_sd = 1e-4,
                            n_iter = 2000, n_chains = 4, warmup = 1000,
                            adapt = FALSE, seed = 1, backend = "cpu")
  expect_equal(mean(fit_adapt$draws[, , 1]), mean(y), tolerance = 0.1)
  # The non-adaptive chain stays frozen near its initialisation, well away
  # from the data mean of 5; the adapted chain reaches the posterior.
  expect_gt(abs(mean(fit_off$draws[, , 1]) - mean(y)),
            abs(mean(fit_adapt$draws[, , 1]) - mean(y)))
  expect_false(is.null(fit_adapt$adaptation))
  expect_null(fit_off$adaptation)
})

test_that("the fit object carries the adaptation book-keeping", {
  set.seed(7)
  y <- rnorm(300, mean = 0, sd = 1)
  m <- gpum_model(~ -((y - mu)^2) / 2, params = "mu", data = "y")
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.1,
                       n_iter = 1000, n_chains = 4, warmup = 500,
                       seed = 1, backend = "cpu")
  ad <- fit$adaptation
  expect_named(ad, c("final_proposal_sd", "final_scales", "n_batches",
                     "batch_sizes", "accept_history"))
  expect_equal(dim(ad$final_proposal_sd), c(4L, 1L))
  expect_length(ad$final_scales, 4L)
  expect_equal(sum(ad$batch_sizes), 500L)        # warmup ran in full
  expect_equal(dim(ad$accept_history), c(4L, ad$n_batches))
  expect_true(all(is.finite(ad$accept_history)))
})
