# Analysis of the reduced experiment run.
#
# Reads benchmark/results/reduced_m1/results.csv, applies the H1 correctness
# gate, summarises ESS per second per cell with bootstrap confidence
# intervals, writes the per-cell table, and draws the performance figure
# embedded in the README.
#
# Usage: Rscript benchmark/analyze.R

results_csv <- "benchmark/results/reduced_m1/results.csv"
fig_dir <- "man/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

r <- utils::read.csv(results_csv, stringsAsFactors = FALSE)
co <- r[r$outcome == "completed", ]

# Bootstrap 95 percent confidence interval of the median.
boot_ci <- function(x, b = 3000) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) return(c(NA_real_, NA_real_))
  meds <- replicate(b, stats::median(sample(x, replace = TRUE)))
  unname(stats::quantile(meds, c(0.025, 0.975)))
}

# Per-cell summary: median ESS/s and its bootstrap CI.
key <- interaction(co$backend, co$N, co$C, drop = TRUE)
cell_tab <- do.call(rbind, lapply(split(co, key), function(d) {
  ci <- boot_ci(d$ess_per_sec)
  data.frame(
    backend = d$backend[1], N = d$N[1], C = d$C[1], reps = nrow(d),
    ess_per_sec_median = stats::median(d$ess_per_sec),
    ci_lo = ci[1], ci_hi = ci[2],
    rhat_max = max(d$rhat, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
cell_tab <- cell_tab[order(cell_tab$backend, as.numeric(cell_tab$N),
                           cell_tab$C), ]
# The per-cell summary is committed (the raw per-run results are not).
utils::write.csv(cell_tab, "benchmark/reduced_m1_cell_summary.csv",
                 row.names = FALSE)

# H1 correctness gate: Holm-Bonferroni over each backend's family of KS tests.
cat("=== H1 correctness gate (Holm-Bonferroni, family alpha 0.05) ===\n")
for (b in sort(unique(co$backend))) {
  pv <- co$ks_pvalue[co$backend == b]
  pv <- pv[is.finite(pv)]
  survives <- if (length(pv)) sum(p.adjust(pv, "holm") < 0.05) else 0L
  cat(sprintf("  %-22s %3d tests, %d rejection(s) survive correction -> %s\n",
              b, length(pv), survives,
              if (survives == 0L) "supported" else "FLAG"))
}
cat(sprintf("\nR-hat across all completed runs: median %.4f, max %.4f\n",
            stats::median(co$rhat, na.rm = TRUE),
            max(co$rhat, na.rm = TRUE)))

# Performance figure: two panels at N = 1e3. Left, one chain (C = 1), the
# head-to-head regime where every backend completes. Right, many chains
# (C = 4096), the regime the package targets. The contrast is the honest
# story: with one chain the GPU does not help; with many chains gpumetropolis
# pulls far ahead and the competitors mostly cannot complete the cell.
backends <- c("gpumetropolis-cuda", "gpumetropolis-vulkan",
              "gpumetropolis-cpu", "nimble", "mcmc", "MCMCpack",
              "Stan-cmdstanr", "BayesianTools")
backends <- backends[backends %in% cell_tab$backend]
ess_at <- function(nn, cc) {
  vapply(backends, function(bk) {
    v <- cell_tab$ess_per_sec_median[cell_tab$backend == bk &
      as.numeric(cell_tab$N) == nn & cell_tab$C == cc]
    if (length(v)) v[1] else NA_real_
  }, numeric(1))
}

png(file.path(fig_dir, "benchmark_ess_per_sec.png"), width = 1040,
    height = 560, res = 110)
op <- par(mfrow = c(1, 2), mar = c(9, 5, 4, 1))
cols <- ifelse(grepl("^gpumetropolis", backends), "#1b7837", "#999999")
for (cc in c(1L, 4096L)) {
  v <- ess_at(1000, cc)
  barplot(v, log = "y", col = cols, border = NA, las = 2,
          ylab = "median ESS / second (log scale)",
          main = sprintf("N = 1e3, C = %d %s", cc,
                          if (cc == 1L) "(one chain)" else "(many chains)"))
}
par(op)
invisible(dev.off())
cat(sprintf("\nfigure written to %s/benchmark_ess_per_sec.png\n", fig_dir))

# Honest verdict: the ratio in each regime.
comp <- c("nimble", "mcmc", "MCMCpack", "Stan-cmdstanr", "BayesianTools")
cat("\n=== ESS/s, gpumetropolis-cuda vs best competitor ===\n")
for (nn in c(1000, 100000)) {
  for (cc in c(1L, 64L, 4096L)) {
    e <- ess_at(nn, cc)
    cuda <- e["gpumetropolis-cuda"]
    cv <- e[comp]
    cv <- cv[is.finite(cv)]
    best <- if (length(cv)) max(cv) else NA_real_
    verdict <- if (!is.finite(cuda)) {
      "cuda cell not completed"
    } else if (!is.finite(best)) {
      "no competitor completed this cell"
    } else {
      sprintf("ratio %.2fx", cuda / best)
    }
    cat(sprintf("  N=%-7s C=%-5d cuda=%-10s best competitor=%-10s %s\n",
                format(nn, scientific = TRUE), cc,
                if (is.finite(cuda)) sprintf("%.0f", cuda) else "-",
                if (is.finite(best)) sprintf("%.0f", best) else "-",
                verdict))
  }
}
