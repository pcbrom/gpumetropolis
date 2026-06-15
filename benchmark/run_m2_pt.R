# Focused M2 re-run that adds the parallel-tempering variant of
# gpumetropolis (v0.3.0) to the M2 cell of the M2-M4 factorial. The aim
# is to record the H1 verdict and ESS/s of method = "pt" against the
# M2-RWM gpumetropolis-cpu baseline and the strongest M2 competitor
# (nimble), at a single representative cell. The full factorial re-run
# stays in `run_full_m234.R`; this script keeps the M2-PT entry that
# the v0.9 amendment of `EXPERIMENT_PROTOCOL.md` records.
#
# Cell: N = 400 observations, C = 8 chains, 4000 iterations per chain,
# half discarded as warmup, R = 20 replications. Honest, light: a
# focused signal rather than the full grid.

suppressMessages({
  library(gpumetropolis)
})
source("benchmark/harness_util.R")
source("benchmark/adapters/cpu_adapters_v2.R")
source("benchmark/models/m2_bimodal.R")

run_m2_cell <- function(adapter_name, spec, N, n_chains, n_iter, reps,
                        seed_base = 20260200L) {
  adapter <- adapters_v2[[adapter_name]]
  if (is.null(adapter)) {
    stop("adapter '", adapter_name, "' is not registered.", call. = FALSE)
  }
  rows <- vector("list", reps)
  for (r in seq_len(reps)) {
    data <- spec$make_data(N, replication = r)
    seed <- seed_base + 10000L * r
    res <- adapter(spec, data, n_iter = n_iter, n_chains = n_chains,
                   seed = seed)
    gate <- h1_gate_ms(res$draws,
                       ref_fn = function(n) spec$ref_sample(data, n))
    ess <- mean_ess(res$draws)
    rows[[r]] <- data.frame(
      adapter = adapter_name,
      rep = r,
      N = N,
      n_chains = n_chains,
      n_iter = n_iter,
      time_sec = res$time_sec,
      ks_stat = gate$ks_stat,
      ks_pvalue = gate$ks_pvalue,
      rhat = gate$rhat,
      ess = ess,
      ess_per_sec = ess / res$time_sec,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

m2_spec <- m2_spec
cell_results <- list()
for (ad in c("gpumetropolis-cpu", "gpumetropolis-cpu-pt", "nimble")) {
  cat(sprintf("\n=== M2 cell, adapter %s ===\n", ad))
  cell_results[[ad]] <- run_m2_cell(ad, m2_spec,
                                    N = 400L, n_chains = 8L,
                                    n_iter = 4000L, reps = 20L)
}
m2_results <- do.call(rbind, cell_results)

summary_table <- aggregate(
  cbind(ks_stat, ks_pvalue, rhat, ess, ess_per_sec, time_sec) ~ adapter,
  data = m2_results, FUN = function(x) c(mean = mean(x), sd = stats::sd(x))
)
print(summary_table)

# Record the "mode coverage" of the cold chain: an honest per-replication
# count of how many of the two modes (positive and negative) appeared in
# the post-warmup draws. RWM with overdispersed starts often shows mode
# coverage equal to the share of chains that started near each mode; PT
# pulls coverage to 1 (both modes) for every chain.
mode_coverage <- function(draws) {
  pooled <- as.vector(draws)
  any_pos <- any(pooled > 1.5)
  any_neg <- any(pooled < -1.5)
  as.integer(any_pos) + as.integer(any_neg)
}
m2_results$mode_coverage <- vapply(seq_len(nrow(m2_results)), function(i) {
  ad <- m2_results$adapter[i]
  r <- m2_results$rep[i]
  mode_coverage(cell_results[[ad]][r, ])
}, integer(1L))
cat("\nMode coverage by adapter (mean of 2 = both modes seen):\n")
print(tapply(m2_results$mode_coverage, m2_results$adapter, mean))

out_dir <- file.path("benchmark", "results", "m2_pt")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_file <- file.path(out_dir,
                      sprintf("m2_pt_%s.csv", format(Sys.Date(), "%Y%m%d")))
utils::write.csv(m2_results, out_file, row.names = FALSE)
cat(sprintf("\nWrote: %s\n", out_file))
