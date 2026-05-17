# Analysis of the M2 to M4 factorial run.
#
# Reads benchmark/results/full_m234/results.csv, applies the H1 correctness
# gate per model and backend, reports the secondary KS rejection-rate
# diagnostic, summarises ESS per second per cell with bootstrap confidence
# intervals, writes the per-cell table, and draws the performance figure.
#
# Usage: Rscript benchmark/analyze_m234.R

results_csv <- "benchmark/results/full_m234/results.csv"
fig_dir <- "man/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

r <- utils::read.csv(results_csv, stringsAsFactors = FALSE)
co <- r[r$outcome == "completed", ]
models <- c("M2", "M3", "M4")

# Bootstrap 95 percent confidence interval of the median.
boot_ci <- function(x, b = 3000) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) return(c(NA_real_, NA_real_))
  meds <- replicate(b, stats::median(sample(x, replace = TRUE)))
  unname(stats::quantile(meds, c(0.025, 0.975)))
}

# Per-cell summary: median ESS/s and its bootstrap CI.
key <- interaction(co$model, co$backend, co$N, co$C, drop = TRUE)
cell_tab <- do.call(rbind, lapply(split(co, key), function(d) {
  ci <- boot_ci(d$ess_per_sec)
  data.frame(
    model = d$model[1], backend = d$backend[1], N = d$N[1], C = d$C[1],
    reps = nrow(d), ess_per_sec_median = stats::median(d$ess_per_sec),
    ci_lo = ci[1], ci_hi = ci[2],
    rhat_max = max(d$rhat, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
cell_tab <- cell_tab[order(cell_tab$model, cell_tab$backend,
                           as.numeric(cell_tab$N), cell_tab$C), ]
utils::write.csv(cell_tab, "benchmark/full_m234_cell_summary.csv",
                 row.names = FALSE)

# H1 correctness gate per model and backend: Holm-Bonferroni over each
# family of KS tests, with the nominal-level rejection rate as the v0.2
# secondary diagnostic.
cat("=== H1 gate per model (Holm-Bonferroni, family alpha 0.05) ===\n")
for (m in models) {
  cat(sprintf("\n-- %s --\n", m))
  for (b in sort(unique(co$backend))) {
    pv <- co$ks_pvalue[co$model == m & co$backend == b]
    pv <- pv[is.finite(pv)]
    if (!length(pv)) next
    survives <- sum(p.adjust(pv, "holm") < 0.05)
    cat(sprintf("  %-22s %3d tests, %2d survive, reject@0.05 %5.1f%% -> %s\n",
                b, length(pv), survives, 100 * mean(pv < 0.05),
                if (survives == 0L) "supported" else "FLAG"))
  }
}

# R-hat summary per model.
cat("\n=== R-hat per model (completed runs) ===\n")
for (m in models) {
  rh <- co$rhat[co$model == m]
  rh <- rh[is.finite(rh)]
  cat(sprintf("  %s: median %.4f, max %.4f\n", m,
              stats::median(rh), max(rh)))
}

# M2 mode coverage: the fraction of replications whose pooled draws pass the
# KS test against the bimodal reference, as a function of the chain count.
# A single chain cannot cover both modes; coverage improves with more chains.
cat("\n=== M2 mode coverage: KS pass rate by chain count C ===\n")
m2 <- co[co$model == "M2", ]
for (cc in sort(unique(m2$C))) {
  pv <- m2$ks_pvalue[m2$C == cc]
  pv <- pv[is.finite(pv)]
  if (length(pv)) {
    cat(sprintf("  C=%-6d %4d tests, KS pass rate %5.1f%%\n",
                cc, length(pv), 100 * mean(pv >= 0.05)))
  }
}

# ESS/s verdict per model: gpumetropolis-cuda against the best competitor.
comp <- c("nimble", "mcmc", "MCMCpack", "Stan-cmdstanr", "BayesianTools")
ess_at <- function(mm, nn, cc) {
  v <- vapply(sort(unique(cell_tab$backend)), function(bk) {
    x <- cell_tab$ess_per_sec_median[cell_tab$model == mm &
      cell_tab$backend == bk & as.numeric(cell_tab$N) == nn &
      cell_tab$C == cc]
    if (length(x)) x[1] else NA_real_
  }, numeric(1))
  v
}
cat("\n=== ESS/s, gpumetropolis-cuda vs best competitor ===\n")
probe <- list(M2 = list(N = 1000, C = c(1L, 64L, 4096L)),
              M3 = list(N = 1000, C = c(1L, 64L, 4096L)),
              M4 = list(N = 1, C = c(1L, 64L, 4096L)))
for (m in models) {
  for (cc in probe[[m]]$C) {
    e <- ess_at(m, probe[[m]]$N, cc)
    cuda <- e["gpumetropolis-cuda"]
    cv <- e[comp]; cv <- cv[is.finite(cv)]
    best <- if (length(cv)) max(cv) else NA_real_
    verdict <- if (!is.finite(cuda)) {
      "cuda cell not completed"
    } else if (!is.finite(best)) {
      "no competitor completed this cell"
    } else {
      sprintf("ratio %.2fx", cuda / best)
    }
    cat(sprintf("  %s C=%-6d cuda=%-11s best competitor=%-11s %s\n",
                m, cc,
                if (is.finite(cuda)) sprintf("%.0f", cuda) else "-",
                if (is.finite(best)) sprintf("%.0f", best) else "-",
                verdict))
  }
}

# Performance figure: one panel per model, median ESS/s by backend at the
# many-chains cell C = 64.
backends <- c("gpumetropolis-cuda", "gpumetropolis-vulkan",
              "gpumetropolis-cpu", "nimble", "mcmc", "MCMCpack",
              "Stan-cmdstanr", "BayesianTools")
png(file.path(fig_dir, "benchmark_m234_ess_per_sec.png"), width = 1320,
    height = 520, res = 110)
op <- par(mfrow = c(1, 3), mar = c(9, 5, 4, 1))
cols <- ifelse(grepl("^gpumetropolis", backends), "#1b7837", "#999999")
for (m in models) {
  nn <- probe[[m]]$N
  v <- vapply(backends, function(bk) {
    x <- cell_tab$ess_per_sec_median[cell_tab$model == m &
      cell_tab$backend == bk & as.numeric(cell_tab$N) == nn &
      cell_tab$C == 64L]
    if (length(x)) x[1] else NA_real_
  }, numeric(1))
  barplot(v, log = "y", col = cols, border = NA, las = 2,
          ylab = "median ESS / second (log scale)",
          main = sprintf("%s, C = 64 (many chains)", m))
}
par(op)
invisible(dev.off())
cat(sprintf("\nfigure written to %s/benchmark_m234_ess_per_sec.png\n", fig_dir))
