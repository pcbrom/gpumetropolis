# Orchestrator for the time-boxed M2 to M4 factorial run.
#
# Completes the registered factorial of EXPERIMENT_PROTOCOL.md section 5
# beyond M1: M2 the separated bimodal posterior, M3 the heavy-tailed location
# model, M4 the ill-conditioned multivariate Gaussian. M2 and M3 sweep the
# N by C grid as M1 did; M4 is a target with no observed data, so it sweeps
# the chain count C only, at a single nominal N, with the data-size axis
# recorded as not applicable (see benchmark/models/m4_illcond.R).
#
# The structure mirrors benchmark/run_full_m1.R: one process per cell, a
# per-cell wall-clock cap, a global watchdog, incremental CSV writes, and
# cells that exceed the cap recorded as budget-exceeded.
#
# Usage: Rscript benchmark/run_full_m234.R

cell_cap_sec <- 60L
parallel_jobs <- 4L        # capped at 4 cores to keep the host responsive
n_reps <- 15L
watchdog_sec <- 4200L      # 70 minute global watchdog
out_dir <- "benchmark/results/full_m234"

n_levels <- c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
c_levels <- c(1L, 8L, 64L, 512L, 4096L, 32768L)
# M4 is three-dimensional with a long iteration budget; its draw array grows
# as n_iter * C * dim, so the chain count is capped at 4096. At C = 32768 the
# array reaches the tens of gigabytes and exhausts both host and GPU memory.
c_levels_m4 <- c(1L, 8L, 64L, 512L, 4096L)
backends <- c("gpumetropolis-cpu", "gpumetropolis-cuda",
              "gpumetropolis-vulkan", "mcmc", "MCMCpack", "nimble",
              "BayesianTools", "Stan-cmdstanr")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# M2 and M3: the N by C grid. M4: C only, at the nominal N = 1.
grid_nc <- expand.grid(C = c_levels, N = n_levels, backend = backends,
                       model = c("M2", "M3"), KEEP.OUT.ATTRS = FALSE,
                       stringsAsFactors = FALSE)
grid_m4 <- expand.grid(C = c_levels_m4, N = 1, backend = backends,
                       model = "M4", KEEP.OUT.ATTRS = FALSE,
                       stringsAsFactors = FALSE)
grid <- rbind(grid_nc, grid_m4)
# Run small cells first, interleaved across the three models: a cell at a
# small N and C completes in well under the cap, a cell at a large N can hit
# it. Ordering by N then C then model means the fast, informative cells of
# every model finish before the global watchdog, and the expensive large-N
# cells are the ones recorded as budget-exceeded if the box is reached.
grid <- grid[order(grid$N, grid$C, match(grid$model, c("M2", "M3", "M4")),
                    match(grid$backend, backends)), ]
# Recorded exclusion, as in the M1 run: nimble runs N up to 1e4 only.
grid <- grid[!(grid$backend == "nimble" & grid$N >= 1e5), ]
# M4 mixes slowly under the ill-conditioned geometry, so it is given a longer
# iteration budget; it is cheap per iteration as the data is a single value.
# The budget is bounded so the n_iter * C * dim draw array stays in memory.
grid$n_iter <- ifelse(grid$model == "M4", 8000L, 4000L)
grid$cell_id <- seq_len(nrow(grid))

message("pre-compiling the Stan models ...")
for (sf in c("benchmark/models/m2.stan", "benchmark/models/m3.stan",
             "benchmark/models/m4.stan")) {
  invisible(cmdstanr::cmdstan_model(sf))
}

jobs <- data.frame(
  cell_id = grid$cell_id, model = grid$model, backend = grid$backend,
  N = format(grid$N, scientific = FALSE, trim = TRUE), C = grid$C,
  n_reps = n_reps, n_iter = grid$n_iter, out_dir = out_dir,
  stringsAsFactors = FALSE
)
jobs_file <- file.path(out_dir, "jobs.tsv")
utils::write.table(jobs, jobs_file, sep = "\t", row.names = FALSE,
                   col.names = FALSE, quote = FALSE)
message(sprintf("M2-M4 run: %d cells, %d reps each, cell cap %ds, %d parallel",
                nrow(grid), n_reps, cell_cap_sec, parallel_jobs))

t_start <- Sys.time()
cmd <- sprintf(
  paste0("timeout -k 30 %d xargs -a %s -L1 -P %d bash -c ",
         "'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; ",
         "timeout -k 15 %d Rscript benchmark/run_cell_v2.R \"$@\"' _"),
  watchdog_sec, jobs_file, parallel_jobs, cell_cap_sec
)
watchdog_exit <- system(cmd)
system("pkill -f 'cmdstan' 2>/dev/null", ignore.stdout = TRUE,
       ignore.stderr = TRUE)
elapsed_min <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
watchdog_fired <- identical(watchdog_exit, 124L)

files <- list.files(out_dir, pattern = "^[0-9]+\\.csv$", full.names = TRUE)
done <- if (length(files)) {
  do.call(rbind, lapply(files, utils::read.csv, stringsAsFactors = FALSE))
} else {
  NULL
}
done_cells <- if (is.null(done)) integer(0) else unique(done$cell_id)
missing <- grid[!(grid$cell_id %in% done_cells), ]
if (nrow(missing)) {
  miss_rows <- do.call(rbind, lapply(seq_len(nrow(missing)), function(i) {
    data.frame(
      cell_id = missing$cell_id[i], model = missing$model[i],
      backend = missing$backend[i],
      N = format(missing$N[i], scientific = FALSE, trim = TRUE),
      C = missing$C[i], replication = seq_len(n_reps), seed = NA_integer_,
      n_iter = missing$n_iter[i], time_sec = NA_real_, total_ess = NA_real_,
      ess_per_sec = NA_real_, rhat = NA_real_, ks_stat = NA_real_,
      ks_pvalue = NA_real_, outcome = "budget-exceeded",
      error = NA_character_, stringsAsFactors = FALSE
    )
  }))
  results <- rbind(if (is.null(done)) NULL else done[names(miss_rows)],
                   miss_rows)
} else {
  results <- done
}
results <- results[order(results$cell_id, results$replication), ]
utils::write.csv(results, file.path(out_dir, "results.csv"),
                 row.names = FALSE)

cat(sprintf("\nM2-M4 run done in %.1f min (watchdog fired: %s)\n",
            elapsed_min, watchdog_fired))
cat(sprintf("rows: %d | completed %d | budget-exceeded %d | error %d\n",
            nrow(results), sum(results$outcome == "completed"),
            sum(results$outcome == "budget-exceeded"),
            sum(results$outcome == "error")))
