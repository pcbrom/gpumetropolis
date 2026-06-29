# Plotting layer: joint posterior geometry (pairs, credible regions, surface
# with level curves in 2D and 3D) and explanatory figures for the Bayesian
# decision tools (posterior probability of a hypothesis, the ROPE rule, the
# Cramer-Rao reference and the Bayes factor). All base graphics; the bivariate
# density is a self-contained base-R kernel estimate, no external package.

# Highest posterior density threshold: the density value whose super-level set
# carries probability `level`, used to draw a credible region as one contour.
.hpd_threshold <- function(z, level) {
  zz <- sort(as.vector(z), decreasing = TRUE)
  cs <- cumsum(zz) / sum(zz)
  zz[max(1L, which(cs >= level)[1L])]
}

# Normal-reference bandwidth (Scott's rule), the same rule the standard
# two-dimensional kernel density estimate uses.
.bandwidth_nrd <- function(x) {
  r <- stats::quantile(x, c(0.25, 0.75), names = FALSE)
  h <- (r[2L] - r[1L]) / 1.34
  4 * 1.06 * min(sqrt(stats::var(x)), h) * length(x)^(-1 / 5)
}

# Bivariate Gaussian kernel density on a regular grid. A self-contained base-R
# estimate, so the plotting layer carries no external dependency.
.gpum_kde2d <- function(x, y, n = 80L) {
  gx <- seq(min(x), max(x), length.out = n)
  gy <- seq(min(y), max(y), length.out = n)
  h <- c(.bandwidth_nrd(x), .bandwidth_nrd(y)) / 4
  h[h <= 0] <- 1e-6
  nx <- length(x)
  ax <- stats::dnorm(outer(gx, x, "-") / h[1L])
  ay <- stats::dnorm(outer(gy, y, "-") / h[2L])
  z <- tcrossprod(ax, ay) / (nx * h[1L] * h[2L])
  list(x = gx, y = gy, z = z)
}

# Points of a `level` Cramer-Rao ellipse for a parameter pair, or NULL.
.crlb_ellipse <- function(crlb, idx, level) {
  if (is.null(crlb) || !isTRUE(crlb$applicable)) return(NULL)
  S <- crlb$crlb[idx, idx, drop = FALSE]
  ctr <- crlb$at[idx]
  th <- seq(0, 2 * pi, length.out = 100L)
  r <- sqrt(stats::qchisq(level, df = 2))
  pts <- t(ctr + t(chol(S)) %*% rbind(cos(th), sin(th)) * r)
  list(x = pts[, 1L], y = pts[, 2L])
}

#' Credible region of a parameter pair
#'
#' Returns the highest posterior density region of two parameters at level
#' `level`, as contour polygons of the bivariate posterior density. The region
#' is the set the parameters concentrate in, the convergence region of the
#' pair. The result composes onto any plot with `lines()`.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param params Length-two character vector of parameter names.
#' @param level Probability mass of the region. Default 0.95.
#' @param n Grid resolution of the density estimate. Default 80.
#' @return An object of class `gpum_region` with the `contours` (a list of
#'   `x`, `y` polygons), the `params` and the `level`. Plot it directly or add
#'   it to an existing plot with `lines()`.
#' @seealso [gpum_pairs()], [gpum_surface()]
#' @export
gpum_region <- function(fit, params, level = 0.95, n = 80L) {
  if (length(params) != 2L) stop("`params` must name two parameters.",
                                 call. = FALSE)
  x <- .gpum_posterior_vec(fit, params[1L])
  y <- .gpum_posterior_vec(fit, params[2L])
  kd <- .gpum_kde2d(x, y, n)
  thr <- .hpd_threshold(kd$z, level)
  cl <- grDevices::contourLines(kd$x, kd$y, kd$z, levels = thr)
  structure(list(contours = cl, params = params, level = level, kde = kd),
            class = "gpum_region")
}

#' @export
lines.gpum_region <- function(x, ...) {
  for (cc in x$contours) graphics::lines(cc$x, cc$y, ...)
  invisible(x)
}

#' @export
plot.gpum_region <- function(x, ...) {
  plot(x$kde$x, x$kde$y, type = "n",
       xlab = x$params[1L], ylab = x$params[2L],
       main = sprintf("%.0f%% credible region", 100 * x$level))
  lines.gpum_region(x, ...)
  invisible(x)
}

#' Posterior density surface of a parameter pair
#'
#' Plots the bivariate posterior density of two parameters as level curves in
#' two dimensions and as a three-dimensional surface with the credible-region
#' contours projected on the floor. The level curves are the highest posterior
#' density regions at the requested masses.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param params Length-two character vector of parameter names.
#' @param type One of `"both"` (default), `"contour"` or `"persp"`.
#' @param levels Probability masses of the level curves. Default
#'   `c(0.5, 0.8, 0.95)`.
#' @param n Grid resolution. Default 80.
#' @param theta,phi Viewing angles of the 3D surface.
#' @param ... Passed to `persp()`.
#' @return The fit, invisibly.
#' @seealso [gpum_region()], [gpum_pairs()]
#' @export
gpum_surface <- function(fit, params, type = c("both", "contour", "persp"),
                         levels = c(0.5, 0.8, 0.95), n = 80L,
                         theta = 30, phi = 25, ...) {
  type <- match.arg(type)
  x <- .gpum_posterior_vec(fit, params[1L])
  y <- .gpum_posterior_vec(fit, params[2L])
  kd <- .gpum_kde2d(x, y, n)
  thr <- vapply(sort(levels, decreasing = TRUE),
                function(L) .hpd_threshold(kd$z, L), numeric(1))
  if (type == "both") {
    op <- graphics::par(mfrow = c(1L, 2L)); on.exit(graphics::par(op), add = TRUE)
  }
  if (type %in% c("both", "contour")) {
    graphics::image(kd$x, kd$y, kd$z, col = grDevices::hcl.colors(32, "Blues", rev = TRUE),
                    xlab = params[1L], ylab = params[2L],
                    main = "Posterior level curves")
    graphics::contour(kd$x, kd$y, kd$z, levels = thr, add = TRUE,
                      drawlabels = FALSE, lwd = 1.5)
  }
  if (type %in% c("both", "persp")) {
    pmat <- graphics::persp(kd$x, kd$y, kd$z, theta = theta, phi = phi,
                            xlab = params[1L], ylab = params[2L], zlab = "density",
                            col = "lightblue", border = NA, shade = 0.4,
                            ticktype = "detailed", main = "Posterior surface", ...)
    z0 <- min(kd$z)
    for (L in thr) {
      for (cc in grDevices::contourLines(kd$x, kd$y, kd$z, levels = L)) {
        xy <- grDevices::trans3d(cc$x, cc$y, z0, pmat)
        graphics::lines(xy, col = "red", lwd = 1.2)
      }
    }
  }
  invisible(fit)
}

#' Pairs plot of the joint posterior
#'
#' A matrix of panels over the parameters: the marginal posterior density with
#' its highest density interval on the diagonal, and the bivariate posterior
#' with its credible-region contour off the diagonal. When a `gpum_crlb` object
#' is supplied and applicable, the Cramer-Rao ellipse is overlaid on the
#' off-diagonal panels, so the recovered convergence region can be read against
#' the information-bound reference.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param crlb Optional `gpum_crlb` object for the ellipse overlay.
#' @param level Mass of the highest density interval and credible region.
#'   Default 0.95.
#' @param max_points Cap on plotted draws per panel. Default 3000.
#' @param n Grid resolution of the bivariate density. Default 60.
#' @param ... Unused.
#' @return The fit, invisibly.
#' @seealso [gpum_region()], [gpum_surface()], [gpum_crlb()]
#' @export
gpum_pairs <- function(fit, crlb = NULL, level = 0.95, max_points = 3000L,
                       n = 60L, ...) {
  P <- .gpum_posterior_matrix(fit)
  nm <- fit$model$params
  np <- ncol(P)
  if (nrow(P) > max_points) {
    P <- P[round(seq(1, nrow(P), length.out = max_points)), , drop = FALSE]
  }
  op <- graphics::par(mfrow = c(np, np), mar = c(3, 3, 1.5, 1),
                      mgp = c(1.8, 0.5, 0))
  on.exit(graphics::par(op), add = TRUE)
  for (i in seq_len(np)) {
    for (j in seq_len(np)) {
      if (i == j) {
        dd <- stats::density(P[, i])
        plot(dd, main = nm[i], xlab = "", ylab = "", lwd = 2)
        h <- hdi(P[, i], level)
        graphics::abline(v = h, col = "blue", lty = 2)
      } else {
        plot(P[, j], P[, i], pch = ".", col = "grey55",
             xlab = nm[j], ylab = nm[i], main = "")
        kd <- .gpum_kde2d(P[, j], P[, i], n)
        thr <- .hpd_threshold(kd$z, level)
        graphics::contour(kd$x, kd$y, kd$z, levels = thr, add = TRUE,
                          drawlabels = FALSE, lwd = 1.5)
        el <- .crlb_ellipse(crlb, c(j, i), level)
        if (!is.null(el)) graphics::lines(el$x, el$y, col = "red", lty = 2,
                                          lwd = 1.5)
      }
    }
  }
  invisible(fit)
}

#' @export
plot.gpum_hypothesis <- function(x, ...) {
  d <- x$draws
  dens <- stats::density(d)
  plot(dens, lwd = 2, xlab = x$parameter, ylab = "posterior density",
       main = sprintf("P(%g < %s < %g) = %.3f",
                      x$lower, x$parameter, x$upper, x$prob))
  lo <- max(x$lower, min(dens$x))
  up <- min(x$upper, max(dens$x))
  sel <- dens$x >= lo & dens$x <= up
  graphics::polygon(c(lo, dens$x[sel], up), c(0, dens$y[sel], 0),
                    col = grDevices::rgb(0, 0, 1, 0.30), border = NA)
  if (is.finite(x$lower)) graphics::abline(v = x$lower, lty = 2)
  if (is.finite(x$upper)) graphics::abline(v = x$upper, lty = 2)
  graphics::lines(dens, lwd = 2)
  invisible(x)
}

#' @export
plot.gpum_rope <- function(x, ...) {
  d <- x$draws
  dens <- stats::density(d)
  ytop <- max(dens$y)
  # Span both the posterior and the ROPE, so the rule is legible even when the
  # ROPE sits far from the posterior, the case the decision turns on. The top
  # headroom holds the legend clear of the density; the bottom holds the HDI
  # bar below the axis.
  xlim <- range(c(dens$x, x$rope))
  plot(dens, lwd = 2, xlab = x$parameter, ylab = "posterior density",
       main = sprintf("%s: %s", x$parameter, x$decision),
       xlim = xlim, ylim = c(-0.08 * ytop, 1.22 * ytop))
  graphics::rect(x$rope[1L], 0, x$rope[2L], ytop,
                 col = grDevices::rgb(1, 0, 0, 0.12), border = NA)
  graphics::abline(v = x$rope, col = "red", lty = 3)
  graphics::segments(x$hdi[["lower"]], -0.045 * ytop, x$hdi[["upper"]],
                     -0.045 * ytop, lwd = 5, col = "blue")
  graphics::lines(dens, lwd = 2)
  graphics::legend("top", horiz = TRUE, bty = "n", inset = 0.01,
                   legend = c("ROPE", sprintf("%.0f%% HDI", 100 * x$ci),
                              sprintf("in ROPE: %.1f%%", 100 * x$pct_in_rope)),
                   col = c(grDevices::rgb(1, 0, 0, 0.5), "blue", NA),
                   lwd = c(8, 5, NA), seg.len = 1.2)
  invisible(x)
}

#' @export
plot.gpum_crlb <- function(x, ...) {
  if (!isTRUE(x$applicable)) {
    plot.new()
    graphics::text(0.5, 0.5, paste0("Cramer-Rao reference not applicable:\n",
                                    x$note), cex = 0.9)
    return(invisible(x))
  }
  m <- rbind(`posterior sd` = x$posterior_sd, `Cramer-Rao sd` = x$crlb_sd)
  colnames(m) <- x$params
  graphics::barplot(m, beside = TRUE, col = c("grey60", "tomato"),
                    legend.text = rownames(m),
                    args.legend = list(x = "topright", bty = "n"),
                    ylab = "standard deviation",
                    main = "Posterior spread vs Cramer-Rao bound")
  invisible(x)
}

#' Posterior predictive sample from a fit
#'
#' Draws `n` parameter vectors from the posterior (the cold chain under
#' parallel tempering) and, for each, calls `generate()` to simulate from the
#' model, returning the pooled posterior-predictive sample. The package does
#' not yet generate from an arbitrary likelihood on its own; the user supplies
#' the family's one-line simulator, which they already know from the model they
#' declared. An automatic generator from the model itself is the goal of the
#' synthesis release on the roadmap.
#'
#' @param fit A `gpum_fit` from [gpu_metropolis()].
#' @param generate A function of one named parameter vector (one posterior
#'   draw) returning one or more simulated observations. It may close over the
#'   data to simulate a covariate-dependent response.
#' @param n Number of posterior draws to simulate from. Default 4000.
#' @return A numeric vector, the pooled posterior-predictive sample.
#' @seealso [gpum_density_compare()]
#' @export
gpum_ppc <- function(fit, generate, n = 4000L) {
  if (!inherits(fit, "gpum_fit")) {
    stop("`fit` must be a gpum_fit from gpu_metropolis().", call. = FALSE)
  }
  P <- .gpum_posterior_matrix(fit)
  colnames(P) <- fit$model$params
  idx <- sample.int(nrow(P), n, replace = TRUE)
  unlist(lapply(idx, function(k) as.numeric(generate(P[k, ]))))
}

#' Compare an observed distribution with generated ones on one plot
#'
#' Overlays the kernel density of the observed data with the kernel density of
#' one or more generated samples on a single set of axes, the visual check that
#' a fitted model reproduces the data it was fit to. For competing models the
#' better fit, especially in the relevant tail, is read off directly.
#'
#' @param observed Numeric vector of the observed data.
#' @param generated A numeric vector, or a named list of numeric vectors, of
#'   generated or posterior-predictive samples (see [gpum_ppc()]).
#' @param labels Optional character labels for the generated samples; taken
#'   from the names of `generated` when it is a named list.
#' @param main,xlab Plot title and x-axis label.
#' @param colors Optional colours for the generated densities.
#' @param ... Passed to `plot()`.
#' @return Invisibly, a list with the observed and generated density objects.
#' @seealso [gpum_ppc()]
#' @export
gpum_density_compare <- function(observed, generated, labels = NULL,
                                 main = "Observed against generated densities",
                                 xlab = "value", colors = NULL, ...) {
  if (is.numeric(generated)) generated <- list(generated)
  if (is.null(labels)) {
    gl <- names(generated)
    labels <- if (!is.null(gl) && all(nzchar(gl))) gl else {
      paste("generated", seq_along(generated))
    }
  }
  do <- stats::density(observed)
  dg <- lapply(generated, stats::density)
  if (is.null(colors)) {
    colors <- grDevices::hcl.colors(max(1L, length(dg)), "Dark 3")
  }
  lty_gen <- seq_along(dg)
  yl <- c(0, max(do$y, vapply(dg, function(z) max(z$y), numeric(1))))
  xl <- range(do$x, unlist(lapply(dg, function(z) range(z$x))))
  plot(do, lwd = 3, col = "black", main = main, xlab = xlab,
       ylab = "density", ylim = yl, xlim = xl, ...)
  graphics::rug(observed)
  for (i in seq_along(dg)) {
    graphics::lines(dg[[i]], col = colors[i], lwd = 2, lty = lty_gen[i])
  }
  graphics::legend("topright", bty = "n", legend = c("observed", labels),
                   col = c("black", colors), lwd = c(3, rep(2, length(dg))),
                   lty = c(1, lty_gen))
  invisible(list(observed = do, generated = dg))
}

#' @export
plot.gpum_bayes_factor <- function(x, ...) {
  v <- c(model1 = x$log_evidence1, model0 = x$log_evidence0)
  graphics::barplot(v, col = c("steelblue", "grey60"),
                    ylab = "log marginal likelihood",
                    main = sprintf("B10 = %.3g (%s)", x$bf10, x$interpretation))
  invisible(x)
}
