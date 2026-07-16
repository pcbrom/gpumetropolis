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

# Optimal acceptance rate for random-walk Metropolis as a function of the
# dimension. 0.234 is the d -> infinity limit of Roberts-Gelman-Gilks
# (1997); in low dimension the optimum is well above it (0.44 in d = 1,
# about 0.35 in d = 2), and targeting the asymptotic value there leaves
# efficiency on the table. Values for d <= 8 follow the numerical results
# of Gelman, Roberts and Gilks (1996).
.am_target_accept <- function(d) {
  low_d <- c(0.44, 0.35, 0.32, 0.30, 0.28, 0.27, 0.26, 0.25)
  if (d >= 1L && d <= 8L) low_d[d] else 0.234
}

# Robbins-Monro update for the log of the proposal scale. The step size
# vanishes as `n_batches_seen^{-2/3}`, slow enough to keep mixing and
# fast enough to converge inside a finite warmup. The default target is
# the dimension-dependent optimum of `.am_target_accept`.
.am_robbins_monro_scale <- function(scale, accept_rate, batch_idx, d,
                                    target = NULL) {
  if (is.null(target)) {
    target <- .am_target_accept(d)
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
                                   proposal_sd, warmup, seed,
                                   auto_stop = FALSE) {
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
  # 2.38 / sqrt(d) is the optimal scaling of the proposal Cholesky under a
  # Gaussian target (Roberts-Rosenthal 2001); starting there means the
  # Robbins-Monro scalar only has to correct the non-Gaussian residual.
  scales <- rep(2.38 / sqrt(np), n_chains)
  current_init <- init_mat
  current_sd <- proposal_sd
  accept_history <- matrix(NA_real_, nrow = n_chains, ncol = n_batches)

  # Full-covariance proposal: from the second batch on, once Welford has a
  # covariance estimate, the proposal becomes `state + L z` with a per-chain
  # lower-triangular `L = scale * chol(cov)`, so the walk carries the
  # correlations of the target instead of the diagonal alone.
  current_L <- NULL
  L_pool <- NULL
  # The covariance is pooled across chains: every chain targets the same
  # posterior, so one well-fed Welford beats n_chains noisy ones. The
  # Robbins-Monro scalar stays per chain, so each chain still finds its own
  # acceptance.
  pool_state <- .am_init_state(np, sigma_init = proposal_sd[1L, ]^2)
  target <- .am_target_accept(np)
  # The first half of the warmup is treated as transient: chains started
  # away from the mode and mis-scaled proposals leave draws in the Welford
  # accumulators that inflate the covariance forever. At mid-warmup every
  # accumulator restarts, so the covariance the sampling phase inherits is
  # estimated from equilibrium draws only.
  reset_batch <- n_batches %/% 2L
  prev_scales <- scales
  converged_streak <- 0L
  batches_run <- 0L
  warmup_consumed <- 0L

  for (b in seq_len(n_batches)) {
    use_L <- np > 1L && !is.null(current_L)
    res <- rust_call(init_flat = as.numeric(t(current_init)),
                     sd_flat = as.numeric(t(current_sd)),
                     n_iter = batch_sizes[b],
                     seed = seed + b - 1L,
                     proposal_mode = if (use_L) 2L else 0L,
                     proposal_l = if (use_L) current_L else numeric(0))
    draws <- array(res$draws,
                   dim = c(res$n_iter, res$n_chains, res$n_params))
    for (c in seq_len(n_chains)) {
      chain_draws <- if (np == 1L) {
        matrix(draws[, c, ], ncol = 1L)
      } else {
        draws[, c, ]
      }
      states[[c]] <- .am_welford_update_batch(states[[c]], chain_draws)
      pool_state <- .am_welford_update_batch(pool_state, chain_draws)
      scales[c] <- .am_robbins_monro_scale(scales[c], res$accept_rate[c],
                                           batch_idx = if (b > reset_batch) {
                                             b - reset_batch
                                           } else {
                                             b
                                           },
                                           d = np, target = target)
      current_init[c, ] <- as.numeric(draws[res$n_iter, c, ])
      cov_c <- .am_covariance(states[[c]])
      diag_sd <- sqrt(pmax(diag(cov_c), 1e-12))
      current_sd[c, ] <- scales[c] * diag_sd
      accept_history[c, b] <- res$accept_rate[c]
    }
    # Mid-warmup restart of the accumulators (see reset_batch above). The
    # covariance restarts empty so equilibrium draws alone define it; the
    # proposal keeps the pre-reset Cholesky shape (`L_pool` below only
    # refreshes once the new accumulator has data), so no batch falls back
    # to a diagonal proposal. The Robbins-Monro scalar restarts at its
    # theoretical optimum with the gain schedule (via `b - reset_batch`)
    # back at full strength: the scalar tuned against the transient
    # covariance is wrong for the equilibrium one, and the decayed gain
    # could not climb back in the batches that remain.
    if (b == reset_batch) {
      diag_now <- pmax(diag(.am_covariance(pool_state)), 1e-12)
      pool_state <- .am_init_state(np, sigma_init = diag_now)
      states <- lapply(seq_len(n_chains), function(c) {
        .am_init_state(np, sigma_init = diag_now)
      })
      scales <- rep(2.38 / sqrt(np), n_chains)
    }
    if (np > 1L) {
      if (pool_state$n >= 2L) {
        L_pool <- .am_cholesky(.am_covariance(pool_state))
      }
      if (!is.null(L_pool)) {
        Lflat <- numeric(n_chains * np * np)
        for (c in seq_len(n_chains)) {
          Lc <- scales[c] * L_pool
          Lflat[((c - 1L) * np * np + 1L):(c * np * np)] <- as.numeric(t(Lc))
        }
        current_L <- Lflat
      }
    }
    batches_run <- b
    warmup_consumed <- warmup_consumed + batch_sizes[b]

    # Early freeze: the acceptance must sit near the asymptotic target AND
    # the Robbins-Monro scales must have stopped moving, both for two
    # consecutive batches, before the remaining warmup budget is handed to
    # the sampling phase. Acceptance alone freezes too early: the scale can
    # pass through the right acceptance while still drifting, and a frozen
    # half-tuned proposal costs more effective draws than the warmup saves.
    if (isTRUE(auto_stop) && b >= max(10L, reset_batch + 5L)) {
      scale_drift <- max(abs(scales / prev_scales - 1))
      if (mean(abs(res$accept_rate - target)) < 0.05 && scale_drift < 0.03) {
        converged_streak <- converged_streak + 1L
      } else {
        converged_streak <- 0L
      }
      if (converged_streak >= 2L) break
    }
    prev_scales <- scales
  }

  list(
    final_proposal_sd = current_sd,
    final_init = current_init,
    final_scales = scales,
    final_states = states,
    final_L = if (np > 1L) current_L else NULL,
    n_batches = batches_run,
    batch_sizes = batch_sizes[seq_len(batches_run)],
    warmup_used = warmup_consumed,
    accept_history = accept_history[, seq_len(batches_run), drop = FALSE],
    seed_next = seed + batches_run
  )
}

# Geometric temperature ladder. The default attaches T = 1 to the first
# chain and grows by a geometric factor so the last chain reaches `t_max`.
# Roberts-Rosenthal (2009) recommend tuning the spacing so the swap
# acceptance between adjacent chains sits near 0.234; the geometric
# default is a reasonable starting point for that.
.pt_default_ladder <- function(n_chains, t_max = 10) {
  if (n_chains <= 1L) return(rep(1.0, n_chains))
  beta <- t_max^(1 / (n_chains - 1L))
  beta^(seq_len(n_chains) - 1L)
}

# Orchestrate a parallel-tempering run. Each batch advances every chain
# at its own temperature; between batches an adjacent-pair swap step is
# proposed. Adaptation, when on, updates per-chain proposal_sd during
# the warmup batches only, and freezes at the end of warmup so the
# sampling phase is stationary.
.pt_orchestrate <- function(rust_call, np, n_chains, init_mat,
                            proposal_sd_mat, temperatures, n_iter,
                            warmup, swap_every, adapt, seed) {
  if (length(temperatures) != n_chains) {
    stop("`temperatures` must have one value per chain.", call. = FALSE)
  }
  if (swap_every < 1L) {
    stop("`swap_every` must be a positive integer.", call. = FALSE)
  }
  total_iters <- n_iter
  swap_every <- as.integer(swap_every)
  warmup <- as.integer(warmup)
  n_batches <- max(1L, total_iters %/% swap_every)
  batch_sizes <- rep(swap_every, n_batches)
  remainder <- total_iters - n_batches * swap_every
  if (remainder > 0L) {
    batch_sizes[n_batches] <- batch_sizes[n_batches] + remainder
  }
  warmup_batches <- max(0L, min(n_batches, warmup %/% swap_every))
  warmup_used <- if (warmup_batches > 0L) {
    sum(batch_sizes[seq_len(warmup_batches)])
  } else {
    0L
  }

  states <- lapply(seq_len(n_chains), function(c) {
    .am_init_state(np, sigma_init = proposal_sd_mat[c, ]^2)
  })
  scales <- rep(1.0, n_chains)
  current_init <- init_mat
  current_sd <- proposal_sd_mat
  current_logpost <- rep(NA_real_, n_chains)

  sampling_iters <- total_iters - warmup_used
  if (sampling_iters < 1L) sampling_iters <- 1L
  draws_kept <- array(NA_real_, dim = c(sampling_iters, n_chains, np))
  accept_acc <- numeric(n_chains)
  iters_acc <- 0L
  kept_filled <- 0L

  swap_pairs <- if (n_chains >= 2L) n_chains - 1L else 0L
  swap_history <- if (swap_pairs > 0L) {
    matrix(NA_real_, nrow = swap_pairs, ncol = n_batches)
  } else {
    matrix(numeric(0), nrow = 0, ncol = 0)
  }
  accept_history <- matrix(NA_real_, nrow = n_chains, ncol = n_batches)

  for (b in seq_len(n_batches)) {
    bsize <- batch_sizes[b]
    res <- rust_call(init_flat = as.numeric(t(current_init)),
                     sd_flat = as.numeric(t(current_sd)),
                     n_iter = bsize,
                     seed = seed + b - 1L,
                     temperatures_flat = temperatures)
    draws_batch <- array(res$draws,
                         dim = c(res$n_iter, res$n_chains, res$n_params))
    current_logpost <- as.numeric(res$last_log_post)
    accept_history[, b] <- res$accept_rate

    in_warmup <- b <= warmup_batches
    if (!in_warmup) {
      start <- kept_filled + 1L
      end <- kept_filled + bsize
      draws_kept[start:end, , ] <- draws_batch
      accept_acc <- accept_acc + bsize * res$accept_rate
      iters_acc <- iters_acc + bsize
      kept_filled <- end
    }

    if (in_warmup && isTRUE(adapt)) {
      for (c in seq_len(n_chains)) {
        chain_draws <- if (np == 1L) {
          matrix(draws_batch[, c, ], ncol = 1L)
        } else {
          draws_batch[, c, ]
        }
        states[[c]] <- .am_welford_update_batch(states[[c]], chain_draws)
        scales[c] <- .am_robbins_monro_scale(scales[c], res$accept_rate[c],
                                             batch_idx = b, d = np)
        cov_c <- .am_covariance(states[[c]])
        diag_sd <- sqrt(pmax(diag(cov_c), 1e-12))
        current_sd[c, ] <- scales[c] * diag_sd
      }
    }

    for (c in seq_len(n_chains)) {
      current_init[c, ] <- as.numeric(draws_batch[res$n_iter, c, ])
    }

    if (swap_pairs > 0L) {
      sweep_order <- if (b %% 2L == 1L) {
        seq.int(1L, swap_pairs, by = 2L)
      } else {
        seq.int(min(2L, swap_pairs), swap_pairs, by = 2L)
      }
      u_vec <- .pt_swap_draws(seed = seed, batch = b,
                              n_draws = length(sweep_order))
      pair_attempts <- integer(swap_pairs)
      pair_accepts <- integer(swap_pairs)
      for (k in seq_along(sweep_order)) {
        i <- sweep_order[k]
        u <- u_vec[k]
        delta <- (current_logpost[i + 1L] - current_logpost[i]) *
          (1 / temperatures[i] - 1 / temperatures[i + 1L])
        pair_attempts[i] <- pair_attempts[i] + 1L
        if (is.finite(delta) && log(u) < delta) {
          tmp_state <- current_init[i, ]
          current_init[i, ] <- current_init[i + 1L, ]
          current_init[i + 1L, ] <- tmp_state
          tmp_lp <- current_logpost[i]
          current_logpost[i] <- current_logpost[i + 1L]
          current_logpost[i + 1L] <- tmp_lp
          pair_accepts[i] <- pair_accepts[i] + 1L
        }
      }
      swap_history[, b] <- ifelse(pair_attempts > 0L,
                                   pair_accepts / pair_attempts,
                                   NA_real_)
    }
  }

  sampling_accept <- if (iters_acc > 0L) {
    accept_acc / iters_acc
  } else {
    res$accept_rate
  }
  list(
    draws = draws_kept[seq_len(kept_filled), , , drop = FALSE],
    accept_rate = sampling_accept,
    final_proposal_sd = current_sd,
    final_scales = scales,
    final_init = current_init,
    n_batches = n_batches,
    batch_sizes = batch_sizes,
    warmup_batches = warmup_batches,
    warmup_used = warmup_used,
    swap_history = swap_history,
    accept_history = accept_history,
    seed_next = seed + n_batches
  )
}

# Orchestrate a Differential Evolution MCMC run (path A: host-orchestrated,
# the difference pool is the population frozen at the start of each batch).
# Each batch advances every chain for `de_every` iterations against the
# snapshot carried in `init_flat`; the kernel draws the difference pairs
# internally and reads the frozen population from that same snapshot, so no
# extra buffer is needed and the chains stay independent within a batch.
# Between batches the snapshot refreshes to the new population. Adaptation,
# when on, tracks the per-chain per-dimension spread during warmup and freezes
# it at the end of warmup, so `de_noise` scales a stationary jitter during
# sampling. Correctness rests on the symmetry of the increment
# `gamma * (Z[a] - Z[b]) + epsilon`: a frozen snapshot is a fixed external
# proposal source, so the acceptance ratio is the density ratio alone.
.de_orchestrate <- function(rust_call, np, n_chains, init_mat,
                            proposal_sd_mat, gamma, de_noise, n_iter,
                            warmup, de_every, adapt, seed) {
  if (de_every < 1L) {
    stop("`de_every` must be a positive integer.", call. = FALSE)
  }
  total_iters <- n_iter
  de_every <- as.integer(de_every)
  warmup <- as.integer(warmup)
  n_batches <- max(1L, total_iters %/% de_every)
  batch_sizes <- rep(de_every, n_batches)
  remainder <- total_iters - n_batches * de_every
  if (remainder > 0L) {
    batch_sizes[n_batches] <- batch_sizes[n_batches] + remainder
  }
  warmup_batches <- max(0L, min(n_batches, warmup %/% de_every))
  warmup_used <- if (warmup_batches > 0L) {
    sum(batch_sizes[seq_len(warmup_batches)])
  } else {
    0L
  }

  states <- lapply(seq_len(n_chains), function(c) {
    .am_init_state(np, sigma_init = proposal_sd_mat[c, ]^2)
  })
  current_init <- init_mat
  current_sd <- proposal_sd_mat

  sampling_iters <- total_iters - warmup_used
  if (sampling_iters < 1L) sampling_iters <- 1L
  draws_kept <- array(NA_real_, dim = c(sampling_iters, n_chains, np))
  accept_acc <- numeric(n_chains)
  iters_acc <- 0L
  kept_filled <- 0L
  accept_history <- matrix(NA_real_, nrow = n_chains, ncol = n_batches)
  # Spread of the population across chains, per dimension, recorded each batch.
  # A column collapsing toward zero signals ensemble degeneracy.
  disp_history <- matrix(NA_real_, nrow = np, ncol = n_batches)

  for (b in seq_len(n_batches)) {
    bsize <- batch_sizes[b]
    res <- rust_call(init_flat = as.numeric(t(current_init)),
                     sd_flat = as.numeric(t(current_sd)),
                     n_iter = bsize,
                     seed = seed + b - 1L,
                     proposal_mode = 1L, gamma = gamma, de_noise = de_noise)
    draws_batch <- array(res$draws,
                         dim = c(res$n_iter, res$n_chains, res$n_params))
    accept_history[, b] <- res$accept_rate

    in_warmup <- b <= warmup_batches
    if (!in_warmup) {
      start <- kept_filled + 1L
      end <- kept_filled + bsize
      draws_kept[start:end, , ] <- draws_batch
      accept_acc <- accept_acc + bsize * res$accept_rate
      iters_acc <- iters_acc + bsize
      kept_filled <- end
    }

    if (in_warmup && isTRUE(adapt)) {
      for (c in seq_len(n_chains)) {
        chain_draws <- if (np == 1L) {
          matrix(draws_batch[, c, ], ncol = 1L)
        } else {
          draws_batch[, c, ]
        }
        states[[c]] <- .am_welford_update_batch(states[[c]], chain_draws)
        cov_c <- .am_covariance(states[[c]])
        current_sd[c, ] <- sqrt(pmax(diag(cov_c), 1e-12))
      }
    }

    for (c in seq_len(n_chains)) {
      current_init[c, ] <- as.numeric(draws_batch[res$n_iter, c, ])
    }
    disp_history[, b] <- apply(current_init, 2L, stats::sd)
  }

  sampling_accept <- if (iters_acc > 0L) {
    accept_acc / iters_acc
  } else {
    res$accept_rate
  }
  list(
    draws = draws_kept[seq_len(kept_filled), , , drop = FALSE],
    accept_rate = sampling_accept,
    final_proposal_sd = current_sd,
    final_init = current_init,
    n_batches = n_batches,
    batch_sizes = batch_sizes,
    warmup_batches = warmup_batches,
    warmup_used = warmup_used,
    accept_history = accept_history,
    disp_history = disp_history,
    seed_next = seed + n_batches
  )
}

# Reproducible uniform draws for the parallel-tempering swap step. The
# draws are derived from `seed` and the batch index, so the same `seed`
# replays exactly. The caller's global RNG state is saved and restored,
# the host loop never disturbs the caller's R stream.
.pt_swap_draws <- function(seed, batch, n_draws) {
  if (n_draws <= 0L) return(numeric(0))
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) {
    get(".Random.seed", envir = .GlobalEnv)
  } else {
    NULL
  }
  mix <- (as.numeric(seed) + as.numeric(batch) * 2654435761) %% 2^31
  set.seed(as.integer(mix))
  u <- stats::runif(n_draws)
  if (!is.null(old_seed)) {
    assign(".Random.seed", old_seed, envir = .GlobalEnv)
  } else if (exists(".Random.seed", envir = .GlobalEnv)) {
    rm(".Random.seed", envir = .GlobalEnv)
  }
  u
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
