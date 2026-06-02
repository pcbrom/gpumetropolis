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
    target <- if (d == 1L) 0.44 else 0.234
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
