# Orchestrator for the registered stage A run, model M1.
#
# Designed to a hard 3 hour ceiling and a 30 GB memory ceiling (protocol
# amendment v0.4):
#   - each (cell, replication) runs as an isolated process, so memory is
#     released on process exit and an external timeout enforces the per-run
#     compute budget by a hard kill;
#   - each job runs under `ulimit -v` so the resident set of the parallel
#     workers cannot exceed the memory ceiling;
#   - jobs are ordered replication-major, so if the global watchdog stops the
#     run early every cell keeps an equal number of completed replications;
#   - the whole sweep is wrapped in a global watchdog that guarantees the run
#     ends within the 3 hour ceiling.
#
# Usage:
#   Rscript benchmark/run_stage_a.R pilot   # small subset, validates the driver
#   Rscript benchmark/run_stage_a.R full    # registered stage A run, model M1

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args) >= 1L) args[1] else "full"

# Run parameters (protocol section 7, amendment v0.4).
n_iter <- 4000L          # tuned so the reference reaches ESS >= 400 per chain
budget_sec <- 30L        # per-run wall-clock budget B
parallel_jobs <- 8L      # parallel workers
n_reps <- 20L            # replications per cell, target
ulimit_kb <- 8000000L    # virtual memory backstop per job against a runaway
watchdog_sec <- 10200L   # global watchdog: 2 h 50 m, guards the 3 h ceiling
out_dir <- "benchmark/results/stageA_m1"
backends <- c("gpumetropolis-CPU", "mcmc", "MCMCpack", "nimble",
              "BayesianTools", "Stan-cmdstanr")

cell_map <- utils::read.csv("benchmark/cell_map.csv", colClasses = c(
  cell_id = "integer", model = "character", backend = "character",
  N = "character", C = "integer", stage = "character"
))
cm <- cell_map[cell_map$model == "M1" & cell_map$backend %in% backends, ]

# Recorded exclusion (protocol section 4): nimble builds an explicit graph node
# per observation, so a model with 1e5 or more likelihood nodes is outside its
# idiomatic design and its build dominates memory and time. nimble runs N up to
# 1e4; the larger N cells are excluded for nimble and recorded.
cm <- cm[!(cm$backend == "nimble" & as.numeric(cm$N) >= 1e5), ]

if (mode == "pilot") {
  cm <- cm[cm$backend %in% c("gpumetropolis-CPU", "Stan-cmdstanr") &
             as.numeric(cm$N) %in% c(1e2, 1e3) & cm$C %in% c(1L, 8L), ]
  n_reps <- 2L
  watchdog_sec <- 600L
  out_dir <- "benchmark/results/stageA_m1_pilot"
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Compile the Stan model once before the run; cmdstanr caches the binary.
message("pre-compiling the Stan model ...")
invisible(cmdstanr::cmdstan_model("benchmark/models/m1.stan"))

# Build the job list, ordered replication-major: replication 1 of every cell,
# then replication 2 of every cell, and so on.
jobs <- do.call(rbind, lapply(seq_len(n_reps), function(rep) {
  data.frame(
    cell_id = cm$cell_id, backend = cm$backend, N = cm$N, C = cm$C,
    replication = rep, n_iter = n_iter, out_dir = out_dir,
    stringsAsFactors = FALSE
  )
}))
jobs_file <- file.path(out_dir, "jobs.tsv")
utils::write.table(jobs, jobs_file, sep = "\t", row.names = FALSE,
                   col.names = FALSE, quote = FALSE)
message(sprintf(paste0("stage A %s: %d cells, %d jobs, budget %ds, %d ",
                       "parallel, watchdog %ds"),
                mode, nrow(cm), nrow(jobs), budget_sec, parallel_jobs,
                watchdog_sec))

# Run all jobs. Each job: virtual-memory cap, then a per-run timeout. The whole
# sweep is wrapped in the global watchdog.
t_start <- Sys.time()
# Each job: single-threaded BLAS (protocol section 7, and it avoids
# oversubscribing the cores), a virtual memory backstop, then the per-run
# timeout. The whole sweep is wrapped in the global watchdog.
cmd <- sprintf(
  paste0("timeout -k 30 %d xargs -a %s -L1 -P %d bash -c ",
         "'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; ulimit -v %d; ",
         "timeout -k 15 %d Rscript benchmark/run_one.R \"$@\"' _"),
  watchdog_sec, jobs_file, parallel_jobs, ulimit_kb, budget_sec
)
watchdog_exit <- system(cmd)
system("pkill -f 'cmdstan' 2>/dev/null", ignore.stdout = TRUE,
       ignore.stderr = TRUE)
elapsed_min <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
watchdog_fired <- identical(watchdog_exit, 124L)

# Collect: read every result file, mark jobs with no file as not-completed.
files <- list.files(out_dir, pattern = "^[0-9]+_[0-9]+\\.csv$",
                    full.names = TRUE)
done <- if (length(files)) {
  do.call(rbind, lapply(files, utils::read.csv, stringsAsFactors = FALSE))
} else {
  NULL
}

job_key <- paste(jobs$cell_id, jobs$replication, sep = "_")
done_key <- if (is.null(done)) character(0) else {
  paste(done$cell_id, done$replication, sep = "_")
}
missing <- jobs[!(job_key %in% done_key), ]
if (nrow(missing)) {
  miss_rows <- data.frame(
    cell_id = missing$cell_id, backend = missing$backend, N = missing$N,
    C = missing$C, replication = missing$replication, seed = NA_integer_,
    n_iter = missing$n_iter, time_sec = NA_real_, total_ess = NA_real_,
    ess_per_sec = NA_real_, rhat = NA_real_, ks_stat = NA_real_,
    ks_pvalue = NA_real_,
    outcome = if (watchdog_fired) "not-completed" else "budget-exceeded",
    error = NA_character_, stringsAsFactors = FALSE
  )
  results <- rbind(
    if (is.null(done)) NULL else done[names(miss_rows)],
    miss_rows
  )
} else {
  results <- done
}
results <- results[order(results$cell_id, results$replication), ]
out_csv <- file.path(out_dir, "results.csv")
utils::write.csv(results, out_csv, row.names = FALSE)

cat(sprintf("\nstage A %s done in %.1f min (watchdog fired: %s)\n",
            mode, elapsed_min, watchdog_fired))
cat(sprintf("jobs: %d total | %d completed | %d budget-exceeded | %d error | %d not-completed\n",
            nrow(results), sum(results$outcome == "completed"),
            sum(results$outcome == "budget-exceeded"),
            sum(results$outcome == "error"),
            sum(results$outcome == "not-completed")))
cat(sprintf("results written to %s\n", out_csv))
