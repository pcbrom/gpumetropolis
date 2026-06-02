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
gpum_diagnose <- function(fit, plot = TRUE, return_data = FALSE) {
  if (!inherits(fit, "gpum_fit")) {
    stop("`fit` must be a gpum_fit from gpu_metropolis().", call. = FALSE)
  }
  draws <- fit$draws
  params <- fit$model$params
  np <- fit$n_params

  stat_rows <- lapply(seq_len(np), function(p) {
    M <- .gpum_param_matrix(draws, p, fit$n_chains)
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
  # and the last-batch acceptance per chain is still below 80% of the
  # asymptotic target. Robbins-Monro converges slowly by design (gamma_t
  # = t^{-2/3}), so a short warmup leaves the chain still climbing at
  # the warmup boundary. Doubling the warmup is the canonical knob.
  adaptation_hint <- NULL
  if (!is.null(fit$adaptation)) {
    ah <- fit$adaptation$accept_history
    if (is.matrix(ah) && ncol(ah) >= 1L) {
      target <- if (np == 1L) 0.44 else 0.234
      last_mean <- mean(ah[, ncol(ah)], na.rm = TRUE)
      if (is.finite(last_mean) && last_mean < 0.8 * target) {
        adaptation_hint <- sprintf(
          paste0("Adaptation still climbing at end of warmup ",
                 "(last-batch accept %.2f, target %.2f). ",
                 "Consider warmup = %d to let the per-chain scale ",
                 "plateau."),
          last_mean, target, 2L * fit$warmup
        )
      }
    }
  }

  cat(sprintf("<gpum_diagnose: %s>\n", verdict[1L]))
  cat(sprintf("  backend %s, chains %d, iterations %d (warmup %d, %s)\n",
              fit$backend, fit$n_chains, fit$n_iter, fit$warmup,
              if (!is.null(fit$adaptation)) "adaptive" else "trim"))
  cat("\n")
  print(format(summary, digits = 4, nsmall = 4), row.names = FALSE)
  cat("\n  ", verdict[2L], "\n", sep = "")
  if (!is.null(adaptation_hint)) {
    cat("  Hint: ", adaptation_hint, "\n", sep = "")
  }

  if (isTRUE(plot)) {
    .gpum_diagnose_plot(fit, params)
  }

  if (isTRUE(return_data)) {
    invisible(list(summary = summary, verdict = verdict,
                   adaptation = fit$adaptation,
                   adaptation_hint = adaptation_hint))
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
.gpum_diagnose_plot <- function(fit, params) {
  np <- fit$n_params
  has_adapt <- !is.null(fit$adaptation)
  n_rows <- np + as.integer(has_adapt)
  op <- graphics::par(mfrow = c(n_rows, 4L),
                      oma = c(0, 0, 1.5, 0),
                      mar = c(3, 3, 2, 1),
                      mgp = c(1.8, 0.5, 0))
  on.exit(graphics::par(op), add = TRUE)

  for (p in seq_len(np)) {
    M <- .gpum_param_matrix(fit$draws, p, fit$n_chains)
    v <- as.vector(M)

    graphics::matplot(M, type = "l", lty = 1,
                      xlab = "iteration", ylab = params[p],
                      main = sprintf("Trace, %s", params[p]))

    dens <- stats::density(v)
    plot(dens, xlab = params[p], ylab = "density",
         main = "Density (pooled)", lwd = 2)

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

  if (has_adapt) {
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
    plot.new()
    plot.window(xlim = c(0, 1), ylim = c(0, 1))
    graphics::text(0.5, 0.5,
                   sprintf("%d warmup batches\nsizes %d to %d",
                           nb, min(fit$adaptation$batch_sizes),
                           max(fit$adaptation$batch_sizes)),
                   cex = 0.9)
  }
}
