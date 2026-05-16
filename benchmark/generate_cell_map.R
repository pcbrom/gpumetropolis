# Generate the frozen cell map for the gpumetropolis benchmark.
#
# The cell map enumerates the full factorial of EXPERIMENT_PROTOCOL.md section 4
# and assigns a stable integer `cell_id` to every cell. The seed scheme of the
# protocol (seed = 10000 * cell_id + replication_index) depends on these ids, so
# the file this script writes is frozen together with the protocol and must not
# be regenerated with a different ordering once results exist.

n_levels <- c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
c_levels <- c(1, 8, 64, 512, 4096, 32768)
models <- c("M1", "M2", "M3", "M4")

# Stage A backends run on the CPU baseline now; Stage B backends need the
# Phase 1 GPU kernel.
backends_stage_a <- c("gpumetropolis-CPU", "MCMCpack", "mcmc", "nimble",
                      "BayesianTools", "greta", "Stan-cmdstanr")
backends_stage_b <- c("gpumetropolis-CUDA", "gpumetropolis-Vulkan-NVIDIA",
                      "gpumetropolis-Vulkan-AMD")
backends <- c(backends_stage_a, backends_stage_b)

# Deterministic nesting: model, then backend, then N, then C.
grid <- expand.grid(
  C = c_levels,
  N = n_levels,
  backend = backends,
  model = models,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
grid <- grid[order(match(grid$model, models),
                    match(grid$backend, backends),
                    grid$N, grid$C), ]

cell_map <- data.frame(
  cell_id = seq_len(nrow(grid)),
  model = grid$model,
  backend = grid$backend,
  N = format(grid$N, scientific = FALSE, trim = TRUE),
  C = grid$C,
  stage = ifelse(grid$backend %in% backends_stage_a, "A", "B"),
  stringsAsFactors = FALSE
)

write.csv(cell_map, "benchmark/cell_map.csv", row.names = FALSE,
          quote = FALSE)
cat(sprintf("wrote benchmark/cell_map.csv: %d cells (%d stage A, %d stage B)\n",
            nrow(cell_map), sum(cell_map$stage == "A"),
            sum(cell_map$stage == "B")))
