# Orchestrator for the registered stage A run, model M1.
#
# Builds the job list from the frozen cell map, runs each (cell, replication)
# as an isolated process under an external timeout that enforces the compute
# budget, then collects the result rows. A job killed by the timeout writes no
# file and is recorded as budget-exceeded.
#
# Usage:
#   Rscript benchmark/run_stage_a.R pilot   # small subset, validates the driver
#   Rscript benchmark/run_stage_a.R full    # registered stage A run, model M1

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args) >= 1L) args[1] else "full"

# Run parameters. n_iter is tuned so the reference sampler reaches ESS >= 400
# per chain on M1 (confirmed in the pilot). The budget B is set to 300 s for
# this run, recorded here per protocol section 7.1, wide enough that model
# compilation does not consume the cap.
n_iter <- 4000L
budget_sec <- 300L
parallel_jobs <- 12L
n_reps <- 40L
out_dir <- "benchmark/results/stageA_m1"
backends <- c("gpumetropolis-CPU", "mcmc", "MCMCpack", "nimble",
              "BayesianTools", "Stan-cmdstanr")

cell_map <- utils::read.csv("benchmark/cell_map.csv", colClasses = c(
  cell_id = "integer", model = "character", backend = "character",
  N = "character", C = "integer", stage = "character"
))
cm <- cell_map[cell_map$model == "M1" & cell_map$backend %in% backends, ]

if (mode == "pilot") {
  cm <- cm[cm$backend %in% c("gpumetropolis-CPU", "Stan-cmdstanr") &
             as.numeric(cm$N) %in% c(1e2, 1e3) & cm$C %in% c(1L, 8L), ]
  n_reps <- 2L
  out_dir <- "benchmark/results/stageA_m1_pilot"
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Compile the Stan model once before the run; cmdstanr caches the binary.
message("pre-compiling the Stan model ...")
invisible(cmdstanr::cmdstan_model("benchmark/models/m1.stan"))

# Build the job list: one row per (cell, replication).
jobs <- do.call(rbind, lapply(seq_len(nrow(cm)), function(i) {
  data.frame(
    cell_id = cm$cell_id[i], backend = cm$backend[i], N = cm$N[i],
    C = cm$C[i], replication = seq_len(n_reps), n_iter = n_iter,
    out_dir = out_dir, stringsAsFactors = FALSE
  )
}))
jobs_file <- file.path(out_dir, "jobs.tsv")
utils::write.table(jobs, jobs_file, sep = "\t", row.names = FALSE,
                   col.names = FALSE, quote = FALSE)
message(sprintf("stage A %s: %d cells, %d jobs, budget %ds, %d parallel",
                mode, nrow(cm), nrow(jobs), budget_sec, parallel_jobs))

# Run all jobs: one process per job, external timeout for the hard budget.
t_start <- Sys.time()
cmd <- sprintf(
  paste0("xargs -a %s -L1 -P %d bash -c ",
         "'timeout -k 15 %d Rscript benchmark/run_one.R \"$@\"' _"),
  jobs_file, parallel_jobs, budget_sec
)
system(cmd)
# Reap any Stan child processes orphaned by a timeout kill.
system("pkill -f 'cmdstan' 2>/dev/null", ignore.stdout = TRUE,
       ignore.stderr = TRUE)
elapsed_min <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))

# Collect: read every result file, mark jobs with no file as budget-exceeded.
files <- list.files(out_dir, pattern = "^[0-9]+_[0-9]+\\.csv$",
                    full.names = TRUE)
done <- if (length(files)) {
  do.call(rbind, lapply(files, utils::read.csv, stringsAsFactors = FALSE))
} else {
  done <- jobs[0, ]
}

key <- function(d) paste(d$cell_id, d$replication, sep = "_")
missing <- jobs[!(key(jobs) %in% key(done)), ]
if (nrow(missing)) {
  miss_rows <- data.frame(
    cell_id = missing$cell_id, backend = missing$backend, N = missing$N,
    C = missing$C, replication = missing$replication, seed = NA_integer_,
    n_iter = missing$n_iter, time_sec = NA_real_, total_ess = NA_real_,
    ess_per_sec = NA_real_, rhat = NA_real_, ks_stat = NA_real_,
    ks_pvalue = NA_real_, outcome = "budget-exceeded", error = NA_character_,
    stringsAsFactors = FALSE
  )
  results <- rbind(done[names(miss_rows)], miss_rows)
} else {
  results <- done
}
results <- results[order(results$cell_id, results$replication), ]
out_csv <- file.path(out_dir, "results.csv")
utils::write.csv(results, out_csv, row.names = FALSE)

cat(sprintf("\nstage A %s done in %.1f min\n", mode, elapsed_min))
cat(sprintf("jobs: %d total | %d completed | %d budget-exceeded | %d error\n",
            nrow(results), sum(results$outcome == "completed"),
            sum(results$outcome == "budget-exceeded"),
            sum(results$outcome == "error")))
cat(sprintf("results written to %s\n", out_csv))
