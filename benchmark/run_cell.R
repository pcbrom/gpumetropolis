# Run one whole cell of the experiment: every replication of one
# (backend, N, C) combination, inside a single process.
#
# One process per cell, not per replication, for a reason: a GPU backend pays
# CUDA or wgpu initialisation and the CubeCL kernel JIT compilation once per
# process. Batching the replications amortises that cost. A warmup call is run
# first and discarded, so it absorbs the initialisation and the JIT; the timed
# replications then measure sampling only.
#
# Each replication's row is written as soon as it finishes, so a process killed
# by the external timeout still leaves the completed replications on disk.
#
# Usage: Rscript benchmark/run_cell.R <cell_id> <backend> <N> <C> <n_reps> \
#                                     <n_iter> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
cell_id <- as.integer(args[1])
backend <- args[2]
N <- as.numeric(args[3])
C <- as.integer(args[4])
n_reps <- as.integer(args[5])
n_iter <- as.integer(args[6])
out_dir <- args[7]

suppressPackageStartupMessages(library(coda))
source("benchmark/models/m1_gaussian_mean.R")
source("benchmark/adapters/cpu_adapters.R")
source("benchmark/harness_util.R")

adapter <- cpu_adapters[[backend]]
out_file <- file.path(out_dir, sprintf("%d.csv", cell_id))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Warmup: one short run, discarded, to pay backend initialisation and the
# kernel JIT before the timed replications.
invisible(tryCatch(
  adapter(m1_spec, m1_spec$make_data(N, 1L), 64L, 2L, 1L),
  error = function(e) NULL
))

rows <- vector("list", n_reps)
for (rep in seq_len(n_reps)) {
  seed <- 10000L * cell_id + rep
  data <- m1_spec$make_data(N, rep)
  truth <- m1_spec$truth(data)
  rows[[rep]] <- tryCatch({
    out <- adapter(m1_spec, data, n_iter, C, seed)
    gate <- h1_gate(out$draws, truth)
    ess <- total_ess(out$draws)
    tsec <- out$time_sec
    data.frame(
      cell_id = cell_id, backend = backend,
      N = format(N, scientific = FALSE, trim = TRUE), C = C,
      replication = rep, seed = seed, n_iter = n_iter, time_sec = tsec,
      total_ess = ess, ess_per_sec = ess / tsec, rhat = gate$rhat,
      ks_stat = gate$ks_stat, ks_pvalue = gate$ks_pvalue,
      outcome = "completed", error = NA_character_, stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      cell_id = cell_id, backend = backend,
      N = format(N, scientific = FALSE, trim = TRUE), C = C,
      replication = rep, seed = seed, n_iter = n_iter, time_sec = NA_real_,
      total_ess = NA_real_, ess_per_sec = NA_real_, rhat = NA_real_,
      ks_stat = NA_real_, ks_pvalue = NA_real_, outcome = "error",
      error = conditionMessage(e), stringsAsFactors = FALSE
    )
  })
  # Persist after each replication so a timeout keeps the completed ones.
  utils::write.csv(do.call(rbind, rows[seq_len(rep)]), out_file,
                   row.names = FALSE)
}
