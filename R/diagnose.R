# One-call diagnostic for a gpum_fit.
#
# The function prints a per-parameter summary, states a convergence
# verdict and opens a multi-panel plot per parameter. The convergence
# rule follows the canonical thresholds: split R-hat below 1.01 and
# effective sample size at or above 400 in every parameter (Vehtari
# et al. 2021). When the fit carries adaptation book-keeping, an extra
# panel shows the per-chain acceptance rate by warmup batch.

#' One-call diagnostic: convergence stats, plots and verdict.
#'
#' Produces a per-parameter table (mean, standard deviation, 2.5%, 50%
#' and 97.5% quantiles, split R-hat, effective sample size and Monte
#' Carlo standard error) and a convergence verdict from the asymptotic
#' canonical thresholds (R-hat below 1.01 in every parameter and ESS at
#' or above 400). When `plot = TRUE`, opens a multi-panel plot per
#' parameter showing the trace, the pooled density, the running mean
#' per chain and the pooled autocorrelation; when the fit carries
#' adaptation book-keeping, an extra panel shows the per-chain
#' acceptance rate by warmup batch with the asymptotic optimum drawn as
#' a horizontal reference.
#'
#' @param fit A `gpum_fit` object from [gpu_metropolis()].
#' @param plot Whether to open the diagnostic plot. Default `TRUE`.
#' @param return_data Whether to return the structured stats invisibly.
#'   Default `FALSE`; the function then returns `NULL` invisibly.
#' @param crlb Optional `gpum_crlb` object from [gpum_crlb()]. When supplied
#'   and applicable, the printout adds a posterior-sd against Cramer-Rao-bound
#'   comparison and the density panels overlay the asymptotic-normal reference
#'   from the inverse Fisher information. Default `NULL`.
#'
#' @return When `return_data = TRUE`, a list with `summary` (one row
#'   per parameter), `verdict` (the convergence diagnosis and a
#'   suggested next step), `adaptation` (passed through from
#'   `fit$adaptation`) and `adaptation_hint` (a character string when
#'   the last warmup batch closed below 80% of the asymptotic target
#'   acceptance, `NULL` otherwise). Otherwise `NULL` invisibly.
#'
#' @seealso [gpu_metropolis()], [rhat()], [ess()]
#' @export
gpum_diagnose <- function(fit, plot = TRUE, return_data = FALSE,
                          crlb = NULL) {
  if (!inherits(fit, "gpum_fit")) {
    stop("`fit` must be a gpum_fit from gpu_metropolis().", call. = FALSE)
  }
  if (!is.null(crlb) && !inherits(crlb, "gpum_crlb")) {
    stop("`crlb` must be a gpum_crlb object from gpum_crlb().", call. = FALSE)
  }
  is_pt <- identical(fit$method, "pt")
  is_de <- identical(fit$method, "de")
  # In parallel tempering only the cold chain (T = 1, first column of the
  # draws array) samples the target posterior. The hot chains feed it
  # through swap proposals but are not posterior samples themselves, so
  # the convergence summary collapses to the cold chain.
  draws <- if (is_pt) {
    fit$draws[, 1L, , drop = FALSE]
  } else {
    fit$draws
  }
  diag_n_chains <- if (is_pt) 1L else fit$n_chains
  params <- fit$model$params
  np <- fit$n_params

  stat_rows <- lapply(seq_len(np), function(p) {
    M <- .gpum_param_matrix(draws, p, diag_n_chains)
    v <- as.vector(M)
    q <- stats::quantile(v, c(0.025, 0.5, 0.975), names = FALSE)
    rh <- rhat(M, warmup = 0L)
    es <- ess(M, warmup = 0L)
    mcse <- stats::sd(v) / sqrt(max(es, 1))
    data.frame(
      parameter = params[p],
      mean = mean(v),
      sd = stats::sd(v),
      q2.5 = q[1L],
      q50 = q[2L],
      q97.5 = q[3L],
      Rhat = rh,
      ESS = es,
      MCSE = mcse,
      stringsAsFactors = FALSE
    )
  })
  summary <- do.call(rbind, stat_rows)
  rownames(summary) <- NULL

  max_rhat <- max(summary$Rhat, na.rm = TRUE)
  min_ess <- min(summary$ESS, na.rm = TRUE)
  if (max_rhat < 1.01 && min_ess >= 400) {
    verdict <- c("Converged",
                 "R-hat below 1.01 and ESS at or above 400 in every parameter.")
  } else if (max_rhat < 1.05) {
    verdict <- c("Inconclusive",
                 "Increase n_iter to raise the effective sample size.")
  } else {
    verdict <- c("Not converged",
                 "Increase warmup or revise the initialisation.")
  }

  # Adaptation hint, only when the fit carries adaptation book-keeping
  # and the last-batch acceptance sits outside [0.8, 1.25] times the
  # dimension-dependent target. Robbins-Monro converges slowly by design
  # (gamma_t = t^{-2/3}), so a short warmup leaves the acceptance still
  # moving at the warmup boundary, from below (steps too wide) or from
  # above (steps too narrow). Doubling the warmup is the canonical knob.
  adaptation_hint <- NULL
  if (!is.null(fit$adaptation) && !is_de) {
    ah <- fit$adaptation$accept_history
    if (is.matrix(ah) && ncol(ah) >= 1L) {
      target <- .am_target_accept(np)
      last_mean <- mean(ah[, ncol(ah)], na.rm = TRUE)
      if (is.finite(last_mean) &&
          (last_mean < 0.8 * target || last_mean > 1.25 * target)) {
        adaptation_hint <- sprintf(
          paste0("Adaptation had not plateaued at end of warmup ",
                 "(last-batch accept %.2f, target %.2f). ",
                 "Consider warmup = %d to let the per-chain scale ",
                 "plateau."),
          last_mean, target, 2L * fit$warmup
        )
      }
    }
  }

  # Differential Evolution collapse hint: the difference proposal degenerates
  # when the ensemble loses its spread in a dimension, since the difference of
  # two near-identical chains is near zero. When any dimension's final
  # population spread falls below 1% of its widest value over the run, the
  # ensemble has collapsed there and the jitter is carrying the chain alone.
  de_hint <- NULL
  if (is_de && !is.null(fit$adaptation$disp_history)) {
    dh <- fit$adaptation$disp_history
    if (is.matrix(dh) && ncol(dh) >= 1L) {
      final_disp <- dh[, ncol(dh)]
      peak_disp <- apply(dh, 1L, max, na.rm = TRUE)
      collapsed <- which(is.finite(final_disp) & is.finite(peak_disp) &
                         peak_disp > 0 & final_disp < 0.01 * peak_disp)
      if (length(collapsed) >= 1L) {
        de_hint <- sprintf(
          paste0("Ensemble spread collapsed in %s (final %.2e vs peak ",
                 "%.2e). Raise de_noise or n_chains to keep the ",
                 "difference proposal alive."),
          paste(params[collapsed], collapse = ", "),
          final_disp[collapsed[1L]], peak_disp[collapsed[1L]]
        )
      }
    }
  }

  # Parallel tempering swap hint: when any adjacent pair finishes the run
  # below 10% acceptance, the temperature ladder is too aggressive and
  # the cold chain is starving for swaps. The remedy is either a smaller
  # `t_max` so the ladder is denser, or more chains spread across the
  # current range.
  swap_hint <- NULL
  if (is_pt && !is.null(fit$adaptation$swap_history)) {
    sh <- fit$adaptation$swap_history
    if (is.matrix(sh) && ncol(sh) >= 1L) {
      sw_mean <- rowMeans(sh, na.rm = TRUE)
      worst <- which.min(sw_mean)
      if (length(worst) == 1L && is.finite(sw_mean[worst]) &&
          sw_mean[worst] < 0.1) {
        swap_hint <- sprintf(
          paste0("Pair (%d, %d) swaps at %.2f, below 0.1. ",
                 "Consider a denser temperature ladder."),
          worst, worst + 1L, sw_mean[worst]
        )
      }
    }
  }

  method_label <- if (is_pt) {
    "pt, cold chain"
  } else if (is_de) {
    sprintf("de, gamma %.3f", fit$gamma %||% NA_real_)
  } else {
    fit$method %||% "rwm"
  }
  cat(sprintf("<gpum_diagnose: %s>\n", verdict[1L]))
  cat(sprintf("  method %s, backend %s, chains %d, iterations %d (warmup %d, %s)\n",
              method_label, fit$backend, fit$n_chains, fit$n_iter, fit$warmup,
              if (!is.null(fit$adaptation)) "adaptive" else "trim"))
  cat("\n")
  print(format(summary, digits = 4, nsmall = 4), row.names = FALSE)
  cat("\n  ", verdict[2L], "\n", sep = "")
  if (!is.null(adaptation_hint)) {
    cat("  Hint: ", adaptation_hint, "\n", sep = "")
  }
  if (!is.null(swap_hint)) {
    cat("  Hint: ", swap_hint, "\n", sep = "")
  }
  if (!is.null(de_hint)) {
    cat("  Hint: ", de_hint, "\n", sep = "")
  }
  if (!is.null(crlb) && isTRUE(crlb$applicable)) {
    cat("  Cramer-Rao (asymptotic, prior-free) reference:\n")
    for (p in seq_len(np)) {
      cat(sprintf("    %-11s posterior sd %.4g vs bound %.4g (ratio %.2f)\n",
                  params[p], summary$sd[p], crlb$crlb_sd[p],
                  summary$sd[p] / crlb$crlb_sd[p]))
    }
  } else if (!is.null(crlb)) {
    cat("  Cramer-Rao reference not applicable: ", crlb$note, "\n", sep = "")
  }

  if (isTRUE(plot)) {
    .gpum_diagnose_plot(fit, params, crlb = crlb)
  }

  if (isTRUE(return_data)) {
    invisible(list(summary = summary, verdict = verdict,
                   adaptation = fit$adaptation,
                   adaptation_hint = adaptation_hint,
                   swap_hint = swap_hint,
                   de_hint = de_hint))
  } else {
    invisible(NULL)
  }
}

# Extract the chain-by-iteration matrix for parameter `p` from a draws
# array, handling the single-chain edge case where the dimension drops.
.gpum_param_matrix <- function(draws, p, n_chains) {
  M <- draws[, , p]
  if (!is.matrix(M)) M <- matrix(M, ncol = n_chains)
  M
}

# Multi-panel diagnostic plot. One row per parameter, four columns
# (trace, density, running mean, ACF). When the fit has adaptation
# book-keeping, an extra row shows the per-chain acceptance over the
# warmup batches with the asymptotic optimum as a horizontal reference.
.gpum_diagnose_plot <- function(fit, params, crlb = NULL) {
  is_pt <- identical(fit$method, "pt")
  is_de <- identical(fit$method, "de")
  has_crlb <- inherits(crlb, "gpum_crlb") && isTRUE(crlb$applicable)
  np <- fit$n_params
  has_adapt <- !is.null(fit$adaptation)
  has_swap <- is_pt && !is.null(fit$adaptation$swap_history) &&
              is.matrix(fit$adaptation$swap_history) &&
              ncol(fit$adaptation$swap_history) > 0L
  n_rows <- np + as.integer(has_adapt) + as.integer(has_swap)
  op <- graphics::par(mfrow = c(n_rows, 4L),
                      oma = c(0, 0, 1.5, 0),
                      mar = c(3, 3, 2, 1),
                      mgp = c(1.8, 0.5, 0))
  on.exit(graphics::par(op), add = TRUE)

  draws_for_plot <- if (is_pt) {
    fit$draws[, 1L, , drop = FALSE]
  } else {
    fit$draws
  }
  panel_n_chains <- if (is_pt) 1L else fit$n_chains

  for (p in seq_len(np)) {
    M <- .gpum_param_matrix(draws_for_plot, p, panel_n_chains)
    v <- as.vector(M)

    graphics::matplot(M, type = "l", lty = 1,
                      xlab = "iteration", ylab = params[p],
                      main = sprintf("Trace, %s", params[p]))

    dens <- stats::density(v)
    plot(dens, xlab = params[p], ylab = "density",
         main = "Density (pooled)", lwd = 2)
    if (has_crlb) {
      # Asymptotic-normal reference from the inverse Fisher information.
      xs <- seq(min(dens$x), max(dens$x), length.out = 200L)
      graphics::lines(xs, stats::dnorm(xs, crlb$at[p], crlb$crlb_sd[p]),
                      col = "red", lty = 2, lwd = 1.5)
    }

    running <- apply(M, 2L, function(col) cumsum(col) / seq_along(col))
    graphics::matplot(running, type = "l", lty = 1,
                      xlab = "iteration", ylab = "running mean",
                      main = "Running mean")

    ac <- stats::acf(v, lag.max = 50L, plot = FALSE)
    lags <- as.numeric(ac$lag)
    vals <- as.numeric(ac$acf)
    ci <- 1.96 / sqrt(ac$n.used)
    plot(lags, vals, type = "h", lwd = 1.5,
         xlab = "lag", ylab = "ACF",
         main = "ACF (pooled)",
         ylim = c(min(0, min(vals, na.rm = TRUE)), 1))
    graphics::abline(h = 0, col = "grey50")
    graphics::abline(h = c(-ci, ci), col = "blue", lty = 2)
  }

  if (has_adapt && is_de) {
    # Differential Evolution diagnostic row: acceptance per chain across the
    # batches, the ensemble spread per dimension across the batches (the
    # quantity that drives the difference proposal), the final per-chain
    # noise scale and a summary box.
    ah <- fit$adaptation$accept_history
    nb <- ncol(ah)
    graphics::matplot(seq_len(nb), t(ah), type = "l", lty = 1,
                      xlab = "batch", ylab = "accept rate",
                      main = "DE: accept per chain",
                      ylim = c(0, 1))
    dh <- fit$adaptation$disp_history
    graphics::matplot(seq_len(ncol(dh)), t(dh), type = "l", lty = 1,
                      xlab = "batch", ylab = "ensemble spread",
                      main = "DE: ensemble spread per dim")
    graphics::abline(h = 0, col = "grey50")
    fsd <- fit$adaptation$final_proposal_sd
    graphics::matplot(seq_len(nrow(fsd)), fsd, type = "b", lty = 1,
                      xlab = "chain", ylab = "noise scale",
                      main = "DE: final noise scale")
    graphics::plot.new()
    graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1))
    graphics::text(0.5, 0.5,
                   sprintf("gamma = %.3f\nde_noise = %.1e\n%d batches",
                           fit$gamma %||% NA_real_, fit$de_noise %||% NA_real_,
                           nb),
                   cex = 0.9)
  } else if (has_adapt) {
    ah <- fit$adaptation$accept_history
    nb <- ncol(ah)
    target <- if (np == 1L) 0.44 else 0.234
    graphics::matplot(seq_len(nb), t(ah), type = "l", lty = 1,
                      xlab = "warmup batch", ylab = "accept rate",
                      main = "Adaptation: accept per chain",
                      ylim = c(0, 1))
    graphics::abline(h = target, lty = 2, col = "red")
    # Final per-chain scale.
    plot(seq_along(fit$adaptation$final_scales),
         fit$adaptation$final_scales, type = "b",
         xlab = "chain", ylab = "final scale",
         main = "Adaptation: final scale")
    # Final per-chain proposal sd, one curve per dimension.
    fsd <- fit$adaptation$final_proposal_sd
    graphics::matplot(seq_len(nrow(fsd)), fsd, type = "b", lty = 1,
                      xlab = "chain", ylab = "final proposal_sd",
                      main = "Adaptation: final proposal_sd")
    graphics::plot.new()
    graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1))
    graphics::text(0.5, 0.5,
                   sprintf("%d warmup batches\nsizes %d to %d",
                           nb, min(fit$adaptation$batch_sizes),
                           max(fit$adaptation$batch_sizes)),
                   cex = 0.9)
  }

  if (has_swap) {
    sh <- fit$adaptation$swap_history
    nb <- ncol(sh)
    graphics::matplot(seq_len(nb), t(sh), type = "l", lty = 1,
                      xlab = "batch", ylab = "swap accept",
                      main = "PT: swap accept per pair",
                      ylim = c(0, 1))
    graphics::abline(h = 0.234, lty = 2, col = "red")

    sw_mean <- rowMeans(sh, na.rm = TRUE)
    plot(seq_along(sw_mean), sw_mean, type = "b",
         xlab = "pair (c, c+1)", ylab = "mean swap accept",
         main = "PT: mean swap per pair",
         ylim = c(0, 1))
    graphics::abline(h = 0.234, lty = 2, col = "red")

    plot(seq_along(fit$temperatures), fit$temperatures, type = "b",
         log = "y", xlab = "chain", ylab = "temperature (log)",
         main = "PT: temperature ladder")

    graphics::plot.new()
    graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1))
    graphics::text(0.5, 0.5,
                   sprintf("ladder T = %.2f to %.2f\nswap_every = %d",
                           min(fit$temperatures), max(fit$temperatures),
                           fit$swap_every),
                   cex = 0.9)
  }
}
