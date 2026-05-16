# Orchestrator for the reduced experiment run of protocol amendments v0.5/v0.6.
#
# A reduced subset of the factorial, model M1, sized to a 20 to 30 minute
# ceiling: N in {1e3, 1e5, 1e7}, C in {1, 64, 4096}, eight backends, ten
# replications. One process per cell (v0.6): the process runs a discarded
# warmup and then the ten replications, so GPU initialisation and the kernel
# JIT are paid once and amortised. Each cell process is bounded by a per-cell
# wall-clock cap; the whole sweep is bounded by a global watchdog.
#
# Usage: Rscript benchmark/run_reduced.R

n_iter <- 4000L
cell_cap_sec <- 120L      # per-cell wall-clock cap (warmup plus replications)
parallel_jobs <- 10L      # real RAM is controlled by this job count
n_reps <- 10L
watchdog_sec <- 1620L     # 27 minutes, guards the 20-30 minute ceiling
out_dir <- "benchmark/results/reduced_m1"

n_levels <- c(1e3, 1e5, 1e7)
c_levels <- c(1L, 64L, 4096L)
backends <- c("gpumetropolis-cpu", "gpumetropolis-cuda",
              "gpumetropolis-vulkan", "mcmc", "MCMCpack", "nimble",
              "BayesianTools", "Stan-cmdstanr")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Build the cell list and assign a stable cell_id (the seed scheme uses it).
grid <- expand.grid(C = c_levels, N = n_levels, backend = backends,
                     KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
grid <- grid[order(match(grid$backend, backends), grid$N, grid$C), ]
# Recorded exclusion: nimble runs N up to 1e4 only.
grid <- grid[!(grid$backend == "nimble" & grid$N >= 1e5), ]
grid$cell_id <- seq_len(nrow(grid))

message("pre-compiling the Stan model ...")
invisible(cmdstanr::cmdstan_model("benchmark/models/m1.stan"))

# One job per cell.
jobs <- data.frame(
  cell_id = grid$cell_id, backend = grid$backend,
  N = format(grid$N, scientific = FALSE, trim = TRUE), C = grid$C,
  n_reps = n_reps, n_iter = n_iter, out_dir = out_dir,
  stringsAsFactors = FALSE
)
jobs_file <- file.path(out_dir, "jobs.tsv")
utils::write.table(jobs, jobs_file, sep = "\t", row.names = FALSE,
                   col.names = FALSE, quote = FALSE)
message(sprintf("reduced run: %d cells, %d reps each, cell cap %ds, %d parallel",
                nrow(grid), n_reps, cell_cap_sec, parallel_jobs))

t_start <- Sys.time()
# No `ulimit -v`: GPU runtimes reserve very large virtual address space, which
# is not real memory, and a virtual cap kills their thread setup. Real memory
# is bounded by `parallel_jobs`.
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

# Collect: one CSV per completed cell, each with up to n_reps rows. A cell with
# no file ran out of its cap before its first replication finished.
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

cat(sprintf("\nreduced run done in %.1f min (watchdog fired: %s)\n",
            elapsed_min, watchdog_fired))
cat(sprintf("rows: %d | completed %d | budget-exceeded %d | error %d\n",
            nrow(results), sum(results$outcome == "completed"),
            sum(results$outcome == "budget-exceeded"),
            sum(results$outcome == "error")))
