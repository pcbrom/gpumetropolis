# Run one whole cell of the M2 to M4 factorial: every replication of one
# (model, backend, N, C) combination, inside a single process.
#
# One process per cell amortises the GPU initialisation and kernel JIT over
# the replications, exactly as benchmark/run_cell.R does for M1. A warmup call
# is run first and discarded. Each replication's row is written as soon as it
# finishes, so a process killed by the external timeout still leaves the
# completed replications on disk.
#
# Usage: Rscript benchmark/run_cell_v2.R <cell_id> <model> <backend> <N> <C> \
#                                        <n_reps> <n_iter> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
cell_id <- as.integer(args[1])
model <- args[2]
backend <- args[3]
N <- as.numeric(args[4])
C <- as.integer(args[5])
n_reps <- as.integer(args[6])
n_iter <- as.integer(args[7])
out_dir <- args[8]

suppressPackageStartupMessages(library(coda))
model_file <- switch(
  model,
  M2 = "benchmark/models/m2_bimodal.R",
  M3 = "benchmark/models/m3_heavy_tail.R",
  M4 = "benchmark/models/m4_illcond.R",
  stop("unknown model: ", model)
)
source(model_file)
source("benchmark/adapters/cpu_adapters_v2.R")
source("benchmark/harness_util.R")

spec <- get(paste0(tolower(model), "_spec"))
adapter <- cpu_adapters[[backend]]
out_file <- file.path(out_dir, sprintf("%d.csv", cell_id))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Warmup: one short run, discarded, to pay backend initialisation and the
# kernel JIT before the timed replications.
invisible(tryCatch(
  adapter(spec, spec$make_data(N, 1L), 64L, 2L, 1L),
  error = function(e) NULL
))

rows <- vector("list", n_reps)
for (rep in seq_len(n_reps)) {
  seed <- 10000L * cell_id + rep
  data <- spec$make_data(N, rep)
  rows[[rep]] <- tryCatch({
    out <- adapter(spec, data, n_iter, C, seed)
    gate <- h1_gate_ms(out$draws, function(n) spec$ref_sample(data, n))
    ess <- mean_total_ess(out$draws)
    tsec <- out$time_sec
    data.frame(
      cell_id = cell_id, model = model, backend = backend,
      N = format(N, scientific = FALSE, trim = TRUE), C = C,
      replication = rep, seed = seed, n_iter = n_iter, time_sec = tsec,
      total_ess = ess, ess_per_sec = ess / tsec, rhat = gate$rhat,
      ks_stat = gate$ks_stat, ks_pvalue = gate$ks_pvalue,
      outcome = "completed", error = NA_character_, stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      cell_id = cell_id, model = model, backend = backend,
      N = format(N, scientific = FALSE, trim = TRUE), C = C,
      replication = rep, seed = seed, n_iter = n_iter, time_sec = NA_real_,
      total_ess = NA_real_, ess_per_sec = NA_real_, rhat = NA_real_,
      ks_stat = NA_real_, ks_pvalue = NA_real_, outcome = "error",
      error = conditionMessage(e), stringsAsFactors = FALSE
    )
  })
  utils::write.csv(do.call(rbind, rows[seq_len(rep)]), out_file,
                   row.names = FALSE)
}
