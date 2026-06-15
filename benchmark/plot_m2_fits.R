# Companion to run_m2_pt.R: produces two PNGs that make visible the
# difference between the random-walk and the parallel-tempering fits on
# the M2 bimodal target. The pooled density is the textbook trap, the
# trace tells the truth.

suppressMessages({
  library(gpumetropolis)
})
source("benchmark/harness_util.R")
source("benchmark/adapters/cpu_adapters_v2.R")
source("benchmark/models/m2_bimodal.R")

spec    <- m2_spec
N       <- 400L
C       <- 8L
n_iter  <- 4000L
seed    <- 20260200L + 10000L * 11L           # replication 11, same as test
data    <- spec$make_data(N, replication = 11L)

fit_rwm    <- cpu_adapters[["gpumetropolis-cpu"]](spec, data, n_iter, C, seed)
fit_nimble <- cpu_adapters[["nimble"]](spec, data, n_iter, C, seed)

# For PT we bypass the harness adapter (which broadcasts the cold chain
# to all n_chains slots for ESS/R-hat consistency) and call
# gpu_metropolis directly to keep the raw cold-plus-hot trace.
model_pt <- gpumetropolis::gpum_model(spec$gpum_loglik,
                                       params = spec$params,
                                       data   = spec$data_names)
init_pt  <- spec$init(data, C)
fit_pt_raw <- gpumetropolis::gpu_metropolis(
  model_pt, data = stats::setNames(list(data), spec$data_names),
  init = init_pt, proposal_sd = spec$proposal_sd(data),
  n_iter = n_iter, method = "pt", seed = seed, backend = "cpu"
)

draws_rwm    <- fit_rwm$draws[, , 1L]
draws_pt     <- fit_pt_raw$draws[, , 1L]        # (n_keep, n_chains): col 1
                                                 # is the cold chain by
                                                 # construction; cols 2..C
                                                 # are the hot chains
draws_nimble <- fit_nimble$draws[, , 1L]

# Reference truth, with the mode locations from the M2 spec
true_c <- 3
out_dir <- file.path("man", "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Panel 1: pooled densities for all three adapters side by side. Note
# that the three look almost identical when pooled: this is the trap.
png(file.path(out_dir, "m2_pt_densities.png"),
    width = 1200, height = 380, res = 130)
par(mfrow = c(1, 3), mar = c(3.5, 3.5, 2.5, 0.5), mgp = c(2, 0.6, 0))

plot_dens <- function(draws, title) {
  v <- as.vector(draws)
  d <- density(v, n = 512)
  plot(d, main = title, xlab = expression(mu), ylab = "density",
       lwd = 2, col = "black",
       xlim = c(-5, 5), ylim = c(0, 0.45))
  abline(v = c(-true_c, true_c), col = "red", lty = 2)
}

plot_dens(draws_rwm,
          sprintf("gpumetropolis RWM (pooled, R-hat = %.1f)",
                  gpumetropolis::rhat(draws_rwm, warmup = 0)))
plot_dens(matrix(draws_pt[, 1L], ncol = 1L),
          sprintf("gpumetropolis PT (cold chain, R-hat = %.2f)",
                  gpumetropolis::rhat(matrix(draws_pt[, 1L], ncol = 1L),
                                      warmup = 0)))
plot_dens(draws_nimble,
          sprintf("nimble (pooled, R-hat = %.1f)",
                  gpumetropolis::rhat(draws_nimble, warmup = 0)))
dev.off()
cat("Wrote: ", file.path(out_dir, "m2_pt_densities.png"), "\n")

# Panel 2: per-chain traces. This is where the truth shows. RWM and
# nimble have 8 chains that each stay in one mode for the whole run;
# PT's cold chain crosses back and forth and the hot chains move
# freely.
png(file.path(out_dir, "m2_pt_traces.png"),
    width = 1200, height = 360, res = 130)
par(mfrow = c(1, 3), mar = c(3.5, 3.5, 2.5, 0.5), mgp = c(2, 0.6, 0))

plot_trace <- function(draws, title, highlight_cold = FALSE) {
  k <- ncol(draws)
  cols <- rainbow(k, v = 0.85)
  if (highlight_cold) cols[1L] <- "black"
  lwds <- rep(0.8, k)
  if (highlight_cold) lwds[1L] <- 1.4
  matplot(draws, type = "l", lty = 1, col = cols, lwd = lwds,
          xlab = "iteration", ylab = expression(mu),
          main = title, ylim = c(-7, 7))
  abline(h = c(-true_c, true_c), col = "red", lty = 2)
}

plot_trace(draws_rwm, "RWM: 8 chains, no crossing")
plot_trace(draws_pt,
           "PT: cold chain (black) crosses, hot chains (colour) free",
           highlight_cold = TRUE)
plot_trace(draws_nimble, "nimble: 8 chains, no crossing")
dev.off()
cat("Wrote: ", file.path(out_dir, "m2_pt_traces.png"), "\n")
