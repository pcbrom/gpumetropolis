# Observed Fisher information and the Cramer-Rao lower bound, as an optional
# diagnostic reference for the MCMC posterior. The bound is a frequentist,
# asymptotic, prior-free object; under the regularity conditions of the
# Bernstein-von Mises theorem the posterior covariance approaches the inverse
# Fisher information, so a match is a sanity check that the sampler recovered
# the information-bound geometry. The function refuses to report a number when
# those conditions visibly fail, rather than print a misleading one.

#' Observed Fisher information and Cramer-Rao bound for a fit
#'
#' Computes the observed Fisher information of the model log-likelihood at the
#' posterior mean by a central-difference Hessian, inverts it to the
#' Cramer-Rao lower bound on the covariance of an unbiased estimator, and
#' returns it beside the empirical posterior covariance for comparison. This
#' is an optional reference, not the headline of the fit: the bound is a
#' frequentist asymptotic object and is meaningful only where the model is
#' regular. The function applies four guards and reports
#' `applicable = FALSE` with a reason when any fails.
#'
#' Under the Bernstein-von Mises theorem, for a regular model with a flat or
#' weak prior the posterior covariance approaches the inverse Fisher
#' information; an agreement is then a check that the sampler recovered the
#' information-bound geometry. The identification is asymptotic and assumes an
#' interior true parameter, identifiability, a finite non-singular information
#' and a smooth correctly specified likelihood. It does not transfer to
#' multimodal posteriors or to parameters on a boundary, and an informative
#' prior legitimately makes the posterior tighter than the bound.
#'
#' The comparison rests on estimator theory: under the same regularity the
#' posterior mean is a consistent and asymptotically efficient estimator, that
#' is it concentrates on the true parameter and attains the Cramer-Rao bound as
#' the data grow, so the bound is the right yardstick for the posterior spread.
#' Consistency is the weaker and more robust property and can hold where
#' asymptotic efficiency does not; this function reports the bound only where
#' the stronger normal-approximation conditions also hold.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param data The same data passed to [gpu_metropolis()]. Required when the
#'   model has a data term; ignored otherwise.
#' @param at Optional numeric vector, the point at which to evaluate the
#'   information. Defaults to the posterior mean (the cold chain for a
#'   parallel-tempering fit, the pooled chains otherwise).
#' @param rhat_max Convergence guard: when the largest split R-hat over the
#'   parameters exceeds this, the comparison is flagged not applicable, since
#'   the asymptotic-normal reference assumes a unimodal converged posterior.
#'   Default 1.05.
#'
#' @return An object of class `gpum_crlb`: a list with `information` (the
#'   observed Fisher information matrix), `crlb` (its inverse, the bound on the
#'   covariance), `crlb_sd`, `crlb_cor`, the empirical `posterior_cov`,
#'   `posterior_sd` and `posterior_cor`, the evaluation point `at`, a logical
#'   `applicable` and a character `note`. When not applicable the matrix
#'   entries are `NA` and `note` states why.
#'
#' @seealso [gpu_metropolis()], [gpum_diagnose()]
#' @export
gpum_crlb <- function(fit, data = NULL, at = NULL, rhat_max = 1.05) {
  if (!inherits(fit, "gpum_fit")) {
    stop("`fit` must be a gpum_fit from gpu_metropolis().", call. = FALSE)
  }
  model <- fit$model
  np <- fit$n_params
  params <- model$params

  # Posterior draws: the cold chain for PT, the pooled chains otherwise.
  is_pt <- identical(fit$method, "pt")
  pdraws <- if (is_pt) fit$draws[, 1L, , drop = FALSE] else fit$draws
  pchains <- if (is_pt) 1L else fit$n_chains
  post_mat <- vapply(seq_len(np), function(p) {
    as.vector(.gpum_param_matrix(pdraws, p, pchains))
  }, numeric(dim(pdraws)[1L] * pchains))
  post_cov <- stats::cov(post_mat)
  post_sd <- sqrt(diag(post_cov))
  post_cor <- stats::cov2cor(post_cov)
  if (is.null(at)) at <- colMeans(post_mat)

  na_mat <- matrix(NA_real_, np, np)
  not_applicable <- function(note) {
    structure(list(
      information = na_mat, crlb = na_mat, crlb_sd = rep(NA_real_, np),
      crlb_cor = na_mat, posterior_cov = post_cov, posterior_sd = post_sd,
      posterior_cor = post_cor, at = at, params = params,
      applicable = FALSE, note = note
    ), class = "gpum_crlb")
  }

  # Guard 1: a model with no data term carries no likelihood information, so
  # the Cramer-Rao bound is undefined.
  if (model$n_cols == 0L) {
    return(not_applicable(
      "Model has no data term; the likelihood carries no Fisher information."))
  }
  if (is.null(data)) {
    stop("`data` is required for a model with a data term; pass the same ",
         "`data` used in gpu_metropolis().", call. = FALSE)
  }

  # Guard 2: a posterior that has not converged to a single mode breaks the
  # asymptotic-normal reference. Flag on the largest split R-hat.
  rh <- vapply(seq_len(np), function(p) {
    rhat(.gpum_param_matrix(pdraws, p, pchains), warmup = 0L)
  }, numeric(1))
  if (max(rh, na.rm = TRUE) > rhat_max) {
    return(not_applicable(sprintf(
      paste0("Largest R-hat %.3f exceeds %.2f; the asymptotic-normal ",
             "reference assumes a converged unimodal posterior."),
      max(rh, na.rm = TRUE), rhat_max)))
  }

  df <- as.data.frame(data, stringsAsFactors = FALSE)
  mat <- as.matrix(df[, model$data, drop = FALSE])
  n_obs <- nrow(mat)
  data_flat <- as.vector(t(mat))

  # Central-difference Hessian of the log-likelihood at `at`. The step per
  # dimension is relative to the posterior spread, floored so it stays finite
  # when a dimension is sharp.
  h <- pmax(1e-4, 1e-3 * post_sd)
  ll <- function(points) {
    rust_loglik_batch(model$loglik$code, model$loglik$consts, np,
                      as.numeric(data_flat), model$n_cols, n_obs,
                      as.numeric(t(points)))
  }
  # Assemble the stencil: the centre, the per-axis plus/minus steps, and the
  # four-corner points for each off-diagonal pair, then evaluate in one call.
  pts <- list(at)
  idx_diag <- list()
  for (j in seq_len(np)) {
    ep <- at; ep[j] <- ep[j] + h[j]
    em <- at; em[j] <- em[j] - h[j]
    idx_diag[[j]] <- c(length(pts) + 1L, length(pts) + 2L)
    pts <- c(pts, list(ep), list(em))
  }
  idx_off <- list()
  if (np >= 2L) {
    for (j in seq_len(np - 1L)) {
      for (k in (j + 1L):np) {
        pp <- at; pp[j] <- pp[j] + h[j]; pp[k] <- pp[k] + h[k]
        pm <- at; pm[j] <- pm[j] + h[j]; pm[k] <- pm[k] - h[k]
        mp <- at; mp[j] <- mp[j] - h[j]; mp[k] <- mp[k] + h[k]
        mm <- at; mm[j] <- mm[j] - h[j]; mm[k] <- mm[k] - h[k]
        idx_off[[paste(j, k)]] <- length(pts) + 1:4
        pts <- c(pts, list(pp), list(pm), list(mp), list(mm))
      }
    }
  }
  vals <- ll(do.call(rbind, pts))
  f0 <- vals[1L]

  H <- matrix(0, np, np)
  for (j in seq_len(np)) {
    fp <- vals[idx_diag[[j]][1L]]
    fm <- vals[idx_diag[[j]][2L]]
    H[j, j] <- (fp - 2 * f0 + fm) / (h[j]^2)
  }
  if (np >= 2L) {
    for (j in seq_len(np - 1L)) {
      for (k in (j + 1L):np) {
        q <- idx_off[[paste(j, k)]]
        hjk <- (vals[q[1L]] - vals[q[2L]] - vals[q[3L]] + vals[q[4L]]) /
          (4 * h[j] * h[k])
        H[j, k] <- hjk
        H[k, j] <- hjk
      }
    }
  }

  information <- -H
  # Guard 3: the observed information must be positive definite. A negative or
  # near-zero eigenvalue signals a non-regular geometry, a boundary or a
  # numerically unstable Hessian, and the bound is then not meaningful.
  ev <- eigen(information, symmetric = TRUE, only.values = TRUE)$values
  if (any(!is.finite(ev)) || min(ev) <= 0) {
    return(not_applicable(sprintf(
      paste0("Observed information is not positive definite (smallest ",
             "eigenvalue %.2e); the model is not regular at this point or ",
             "the numerical Hessian is unstable."), min(ev))))
  }

  crlb <- solve(information)
  crlb_sd <- sqrt(diag(crlb))
  crlb_cor <- stats::cov2cor(crlb)

  structure(list(
    information = information, crlb = crlb, crlb_sd = crlb_sd,
    crlb_cor = crlb_cor, posterior_cov = post_cov, posterior_sd = post_sd,
    posterior_cor = post_cor, at = at, params = params,
    applicable = TRUE,
    note = paste0("Frequentist asymptotic reference (inverse Fisher ",
                  "information), prior-free. A weak/flat prior is assumed; ",
                  "an informative prior makes the posterior tighter.")
  ), class = "gpum_crlb")
}

#' @export
print.gpum_crlb <- function(x, ...) {
  cat("<gpum_crlb>\n")
  if (!isTRUE(x$applicable)) {
    cat("  Comparison not applicable.\n")
    cat("  ", x$note, "\n", sep = "")
    return(invisible(x))
  }
  tab <- data.frame(
    parameter = x$params,
    posterior_sd = x$posterior_sd,
    cramer_rao_sd = x$crlb_sd,
    ratio = x$posterior_sd / x$crlb_sd,
    stringsAsFactors = FALSE
  )
  print(format(tab, digits = 4, nsmall = 4), row.names = FALSE)
  if (length(x$params) >= 2L) {
    cat(sprintf("\n  posterior correlation %.3f, Cramer-Rao %.3f\n",
                x$posterior_cor[1L, 2L], x$crlb_cor[1L, 2L]))
  }
  cat("\n  ", x$note, "\n", sep = "")
  invisible(x)
}
