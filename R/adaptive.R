# Adaptive Metropolis primitives.
#
# Pure-R numerical primitives for the host-side orchestration of the
# adaptive warmup. The covariance update is Welford's online algorithm
# (Welford 1962), the scale update is the Robbins-Monro step targeting
# the asymptotic optimal acceptance rate (Roberts and Rosenthal 2009),
# and the Cholesky factorisation regularises ill-conditioned estimates
# during the early warmup. The Rust kernel still consumes a diagonal
# proposal in this development cut; the full Cholesky path opens when
# the kernel accepts an `L` matrix per chain.

# Initialise an AM state for a single chain with `d` parameters.
# `sigma_init` is the starting diagonal of Sigma (recycled to length d).
# `scale_init` is the starting scalar multiplier.
.am_init_state <- function(d, sigma_init = 1.0, scale_init = 1.0) {
  list(
    d = as.integer(d),
    n = 0L,
    mean = rep(0.0, d),
    # Sum of squared deviations from the running mean. The covariance is
    # M2 / (n - 1) once n >= 2.
    M2 = matrix(0.0, nrow = d, ncol = d),
    sigma = diag(rep(sigma_init, length.out = d), nrow = d),
    scale = as.numeric(scale_init)
  )
}

# Update the running mean and the cross-product matrix M2 with one new
# sample. The update is numerically stable for sequential streaming and
# matches `cov()` to machine precision once enough samples are seen.
.am_welford_update <- function(state, x) {
  x <- as.numeric(x)
  if (length(x) != state$d) {
    stop(".am_welford_update: dimension mismatch.", call. = FALSE)
  }
  state$n <- state$n + 1L
  delta <- x - state$mean
  state$mean <- state$mean + delta / state$n
  delta2 <- x - state$mean
  state$M2 <- state$M2 + tcrossprod(delta, delta2)
  state
}

# Batch update with a matrix `X` of one row per sample. Equivalent to
# calling `.am_welford_update()` on each row but vectorised.
.am_welford_update_batch <- function(state, X) {
  X <- as.matrix(X)
  if (ncol(X) != state$d) {
    stop(".am_welford_update_batch: dimension mismatch.", call. = FALSE)
  }
  for (i in seq_len(nrow(X))) {
    state <- .am_welford_update(state, X[i, ])
  }
  state
}

# Return the current empirical covariance, or the initial Sigma when
# fewer than two samples have been seen.
.am_covariance <- function(state) {
  if (state$n < 2L) state$sigma else state$M2 / (state$n - 1L)
}

# Robbins-Monro update for the log of the proposal scale. The step size
# vanishes as `n_batches_seen^{-2/3}`, slow enough to keep mixing and
# fast enough to converge inside a finite warmup. Targets 0.234 in
# dimensions >= 2 and 0.44 in dimension 1, the asymptotic optima of
# Roberts-Gelman-Gilks (1997).
.am_robbins_monro_scale <- function(scale, accept_rate, batch_idx, d,
                                    target = NULL) {
  if (is.null(target)) {
    target <- if (d == 1L) 0.44 else 0.234
  }
  gamma <- (as.numeric(batch_idx))^(-2 / 3)
  exp(log(scale) + gamma * (accept_rate - target))
}

# Regularised Cholesky factorisation. The regulariser is `eps` times the
# mean of the diagonal so the perturbation is scale-invariant in the
# units of the parameters. Falls back to the diagonal Cholesky when the
# perturbed matrix is still indefinite, which never happens in practice
# but keeps the contract `length(L) > 0` safe.
.am_cholesky <- function(Sigma, eps = 1e-8) {
  d <- nrow(Sigma)
  reg <- eps * mean(diag(Sigma))
  if (!is.finite(reg) || reg <= 0) reg <- eps
  Sigma_reg <- Sigma + reg * diag(d)
  out <- tryCatch(chol(Sigma_reg), error = function(e) NULL)
  if (is.null(out)) {
    out <- diag(sqrt(pmax(diag(Sigma_reg), reg)), nrow = d)
  }
  t(out)  # return the lower-triangular factor L such that L %*% t(L) = Sigma_reg
}
