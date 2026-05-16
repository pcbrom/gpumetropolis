# Orchestrator for the time-boxed M1 factorial run.
#
# The full 6x6 grid of EXPERIMENT_PROTOCOL.md section 4 for model M1 (N in
# {1e2 .. 1e7}, C in {1 .. 32768}), eight backends, 15 replications, sized to a
# roughly 25 minute box. Finer N and C resolution than the 3x3 reduced run;
# the largest cells (N=1e7) exceed the per-cell cap and are recorded as
# budget-exceeded, as the protocol's budget mechanism prescribes.
#
# Usage: Rscript benchmark/run_full_m1.R

n_iter <- 4000L
cell_cap_sec <- 60L       # per-cell wall-clock cap (warmup plus replications)
parallel_jobs <- 10L
n_reps <- 15L
watchdog_sec <- 1680L     # 28 minute global watchdog
out_dir <- "benchmark/results/full_m1"

n_levels <- c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
c_levels <- c(1L, 8L, 64L, 512L, 4096L, 32768L)
backends <- c("gpumetropolis-cpu", "gpumetropolis-cuda",
              "gpumetropolis-vulkan", "mcmc", "MCMCpack", "nimble",
              "BayesianTools", "Stan-cmdstanr")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

grid <- expand.grid(C = c_levels, N = n_levels, backend = backends,
                     KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
grid <- grid[order(match(grid$backend, backends), grid$N, grid$C), ]
# Recorded exclusion: nimble runs N up to 1e4 only.
grid <- grid[!(grid$backend == "nimble" & grid$N >= 1e5), ]
grid$cell_id <- seq_len(nrow(grid))

message("pre-compiling the Stan model ...")
invisible(cmdstanr::cmdstan_model("benchmark/models/m1.stan"))

jobs <- data.frame(
  cell_id = grid$cell_id, backend = grid$backend,
  N = format(grid$N, scientific = FALSE, trim = TRUE), C = grid$C,
  n_reps = n_reps, n_iter = n_iter, out_dir = out_dir,
  stringsAsFactors = FALSE
)
jobs_file <- file.path(out_dir, "jobs.tsv")
utils::write.table(jobs, jobs_file, sep = "\t", row.names = FALSE,
                   col.names = FALSE, quote = FALSE)
message(sprintf("full M1 run: %d cells, %d reps each, cell cap %ds, %d parallel",
                nrow(grid), n_reps, cell_cap_sec, parallel_jobs))

t_start <- Sys.time()
cmd <- sprintf(
  paste0("timeout -k 30 %d xargs -a %s -L1 -P %d bash -c ",
         "'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; ",
         "timeout -k 15 %d Rscript benchmark/run_cell.R \"$@\"' _"),
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
      cell_id = missing$cell_id[i], backend = missing$backend[i],
      N = format(missing$N[i], scientific = FALSE, trim = TRUE),
      C = missing$C[i], replication = seq_len(n_reps), seed = NA_integer_,
      n_iter = n_iter, time_sec = NA_real_, total_ess = NA_real_,
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

cat(sprintf("\nfull M1 run done in %.1f min (watchdog fired: %s)\n",
            elapsed_min, watchdog_fired))
cat(sprintf("rows: %d | completed %d | budget-exceeded %d | error %d\n",
            nrow(results), sum(results$outcome == "completed"),
            sum(results$outcome == "budget-exceeded"),
            sum(results$outcome == "error")))
