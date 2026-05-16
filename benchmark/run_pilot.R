# Pilot run for stage A of the benchmark.
#
# A small subset of the frozen cell map, run end to end, to validate the
# pipeline before the full stage A: data generation, sampling through each
# adapter, the H1 correctness gate, and ESS per second. Run from the package
# root: Rscript benchmark/run_pilot.R

suppressPackageStartupMessages({
  library(coda)
})
source("benchmark/models/m1_gaussian_mean.R")
source("benchmark/adapters/cpu_adapters.R")

cell_map <- utils::read.csv("benchmark/cell_map.csv", colClasses = c(
  cell_id = "integer", model = "character", backend = "character",
  N = "character", C = "integer", stage = "character"
))

# Total effective sample size, summed over chains, with coda as the single
# uniform estimator (protocol section 6.2).
total_ess <- function(draws) {
  chains <- lapply(seq_len(ncol(draws)), function(ch) coda::mcmc(draws[, ch]))
  sum(coda::effectiveSize(coda::as.mcmc.list(chains)))
}

# Evenly spaced subsample down to k draws.
thin_to <- function(v, k) {
  k <- max(2L, min(as.integer(k), length(v)))
  if (k >= length(v)) return(v)
  v[round(seq(1, length(v), length.out = k))]
}

# H1 correctness gate: two-sample KS of the thinned pooled draws against an
# exact sample from the closed-form posterior, plus split R-hat.
h1_gate <- function(draws, truth, alpha = 0.01) {
  pooled <- as.vector(draws)
  k <- floor(total_ess(draws))
  thinned <- thin_to(pooled, k)
  reference <- stats::rnorm(length(thinned), truth$mean, truth$sd)
  ks <- suppressWarnings(stats::ks.test(thinned, reference))
  rh <- gpumetropolis::rhat(draws, warmup = 0L)
  list(
    ks_stat = unname(ks$statistic),
    ks_pvalue = ks$p.value,
    rhat = rh,
    pass = is.finite(rh) && rh < 1.01 && ks$p.value > alpha
  )
}

run_cell <- function(spec, backend, N, C, replication, cell_id, n_iter) {
  seed <- 10000L * cell_id + replication
  data <- spec$make_data(N, replication)
  truth <- spec$truth(data)
  adapter <- cpu_adapters[[backend]]
  out <- tryCatch({
    res <- adapter(spec, data, n_iter, C, seed)
    gate <- h1_gate(res$draws, truth)
    ess <- total_ess(res$draws)
    data.frame(
      cell_id = cell_id, model = spec$id, backend = backend, N = N, C = C,
      replication = replication, seed = seed, n_iter = n_iter,
      time_sec = res$time_sec, total_ess = ess,
      ess_per_sec = ess / res$time_sec, rhat = gate$rhat,
      ks_stat = gate$ks_stat, ks_pvalue = gate$ks_pvalue,
      h1_pass = gate$pass, error = NA_character_, stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      cell_id = cell_id, model = spec$id, backend = backend, N = N, C = C,
      replication = replication, seed = seed, n_iter = n_iter,
      time_sec = NA_real_, total_ess = NA_real_, ess_per_sec = NA_real_,
      rhat = NA_real_, ks_stat = NA_real_, ks_pvalue = NA_real_,
      h1_pass = FALSE, error = conditionMessage(e), stringsAsFactors = FALSE
    )
  })
  out
}

# Pilot subset: model M1, three CPU backends, small N and C, five replications.
pilot_N <- c(1e2, 1e3, 1e4)
pilot_C <- c(1L, 8L)
pilot_backends <- c("gpumetropolis-CPU", "mcmc", "MCMCpack")
pilot_reps <- 1:5
n_iter <- 4000L

rows <- list()
for (backend in pilot_backends) {
  for (N in pilot_N) {
    for (C in pilot_C) {
      cid <- cell_map$cell_id[cell_map$model == "M1" &
                                cell_map$backend == backend &
                                as.numeric(cell_map$N) == N &
                                cell_map$C == C]
      for (rep in pilot_reps) {
        cat(sprintf("M1 %-18s N=%-8g C=%-3d rep=%d\n", backend, N, C, rep))
        rows[[length(rows) + 1L]] <- run_cell(m1_spec, backend, N, C, rep,
                                              cid, n_iter)
      }
    }
  }
}

results <- do.call(rbind, rows)
dir.create("benchmark/results", showWarnings = FALSE, recursive = TRUE)
utils::write.csv(results, "benchmark/results/pilot_stage_a.csv",
                 row.names = FALSE)

cat("\n=== pilot summary: median ESS/s and H1 pass rate by cell ===\n")
by_cell <- do.call(rbind, by(results, list(results$backend, results$N,
                                           results$C), function(d) {
  data.frame(
    backend = d$backend[1], N = d$N[1], C = d$C[1],
    median_ess_per_sec = round(stats::median(d$ess_per_sec), 1),
    median_time_sec = round(stats::median(d$time_sec), 4),
    h1_pass = sprintf("%d/%d", sum(d$h1_pass), nrow(d)),
    stringsAsFactors = FALSE
  )
}))
by_cell <- by_cell[order(by_cell$N, by_cell$C, by_cell$backend), ]
print(by_cell, row.names = FALSE)
