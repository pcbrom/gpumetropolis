# Run one (cell, replication) of stage A and write its result row.
#
# Invoked as a separate process so an external `timeout` can enforce the
# compute budget of protocol section 7.1 with a hard kill. A run that is killed
# writes no file; the collector then records it as budget-exceeded.
#
# A run that finishes is recorded as completed or error. A run killed by the
# external timeout writes no file and is recorded as budget-exceeded by the
# collector.
#
# Usage: Rscript benchmark/run_one.R <cell_id> <backend> <N> <C> <rep> \
#                                    <n_iter> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
cell_id <- as.integer(args[1])
backend <- args[2]
N <- as.numeric(args[3])
C <- as.integer(args[4])
replication <- as.integer(args[5])
n_iter <- as.integer(args[6])
out_dir <- args[7]

suppressPackageStartupMessages(library(coda))
source("benchmark/models/m1_gaussian_mean.R")
source("benchmark/adapters/cpu_adapters.R")
source("benchmark/harness_util.R")

seed <- 10000L * cell_id + replication
data <- m1_spec$make_data(N, replication)
truth <- m1_spec$truth(data)

res <- tryCatch({
  out <- cpu_adapters[[backend]](m1_spec, data, n_iter, C, seed)
  gate <- h1_gate(out$draws, truth)
  ess <- total_ess(out$draws)
  list(time_sec = out$time_sec, total_ess = ess,
       ess_per_sec = ess / out$time_sec, rhat = gate$rhat,
       ks_stat = gate$ks_stat, ks_pvalue = gate$ks_pvalue,
       outcome = "completed", error = NA_character_)
}, error = function(e) {
  list(time_sec = NA_real_, total_ess = NA_real_, ess_per_sec = NA_real_,
       rhat = NA_real_, ks_stat = NA_real_, ks_pvalue = NA_real_,
       outcome = "error", error = conditionMessage(e))
})

row <- data.frame(
  cell_id = cell_id, backend = backend,
  N = format(N, scientific = FALSE, trim = TRUE), C = C,
  replication = replication, seed = seed, n_iter = n_iter,
  time_sec = res$time_sec, total_ess = res$total_ess,
  ess_per_sec = res$ess_per_sec, rhat = res$rhat, ks_stat = res$ks_stat,
  ks_pvalue = res$ks_pvalue, outcome = res$outcome, error = res$error,
  stringsAsFactors = FALSE
)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
utils::write.csv(row, file.path(out_dir, sprintf("%d_%d.csv", cell_id,
                                                 replication)),
                 row.names = FALSE)
