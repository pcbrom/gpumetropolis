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
# Orchestrate the batched adaptive warmup. The Rust kernel is called once
# per batch with the current per-chain proposal scale; between batches,
# Welford updates the per-chain running covariance and Robbins-Monro
# updates the per-chain scalar so the acceptance moves toward the optimal
# asymptotic rate. Returns the final per-chain proposal_sd, the final
# per-chain state to seed the sampling phase, and book-keeping for the
# fit object. The orchestrator stays at the diagonal level: the proposal
# is `scale_c * sqrt(diag(cov_c))` per dimension; the off-diagonal
# Cholesky path opens when the kernel consumes an `L` matrix per chain.
.am_orchestrate_warmup <- function(rust_call, np, n_chains, init_mat,
                                   proposal_sd, warmup, seed) {
  if (warmup <= 0L) {
    stop(".am_orchestrate_warmup needs warmup > 0.", call. = FALSE)
  }

  # Heuristic for the batch decomposition: a batch of ~50 iterations gives
  # enough acceptance stats to update the scale; n_batches caps at 40 so
  # the host-side round trips stay modest. n_batches is then refined so
  # n_batches * k is exactly the warmup that runs.
  n_batches <- min(40L, max(10L, warmup %/% 50L))
  if (n_batches > warmup) n_batches <- warmup
  k <- warmup %/% n_batches
  # The last batch absorbs the remainder so the warmup that runs is
  # exactly what the user asked for.
  batch_sizes <- rep(k, n_batches)
  batch_sizes[n_batches] <- batch_sizes[n_batches] + (warmup - n_batches * k)
  warmup_used <- warmup

  states <- lapply(seq_len(n_chains), function(c) {
    .am_init_state(np, sigma_init = proposal_sd[c, ]^2)
  })
  scales <- rep(1.0, n_chains)
  current_init <- init_mat
  current_sd <- proposal_sd
  accept_history <- matrix(NA_real_, nrow = n_chains, ncol = n_batches)

  for (b in seq_len(n_batches)) {
    res <- rust_call(init_flat = as.numeric(t(current_init)),
                     sd_flat = as.numeric(t(current_sd)),
                     n_iter = batch_sizes[b],
                     seed = seed + b - 1L)
    draws <- array(res$draws,
                   dim = c(res$n_iter, res$n_chains, res$n_params))
    for (c in seq_len(n_chains)) {
      chain_draws <- if (np == 1L) {
        matrix(draws[, c, ], ncol = 1L)
      } else {
        draws[, c, ]
      }
      states[[c]] <- .am_welford_update_batch(states[[c]], chain_draws)
      scales[c] <- .am_robbins_monro_scale(scales[c], res$accept_rate[c],
                                           batch_idx = b, d = np)
      current_init[c, ] <- as.numeric(draws[res$n_iter, c, ])
      cov_c <- .am_covariance(states[[c]])
      diag_sd <- sqrt(pmax(diag(cov_c), 1e-12))
      current_sd[c, ] <- scales[c] * diag_sd
      accept_history[c, b] <- res$accept_rate[c]
    }
  }

  list(
    final_proposal_sd = current_sd,
    final_init = current_init,
    final_scales = scales,
    final_states = states,
    n_batches = n_batches,
    batch_sizes = batch_sizes,
    warmup_used = warmup_used,
    accept_history = accept_history,
    seed_next = seed + n_batches
  )
}

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
