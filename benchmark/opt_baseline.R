# v0.5.0 optimisation baseline: the single-chain-regime head-to-heads on the
# three applied cases, measured reproducibly. Run before and after each
# optimisation lever; the CSV is the audit trail of the milestone.
#
# Victory criterion (recorded in ROADMAP.md): (a) beat the generic samplers
# (mcmc, BayesianTools) in ESS/s everywhere; (b) beat or tie Stan on the
# non-conjugate applied case (Gumbel); (c) against conjugate specialists
# (MCMCregress Gibbs) close the gap as far as a generic sampler can, the
# many-chains regime being where this package already wins.
#
# Usage: Rscript benchmark/opt_baseline.R <tag>
# Writes benchmark/opt_results_<tag>.csv

suppressMessages({library(gpumetropolis); library(coda)})
args <- commandArgs(trailingOnly = TRUE)
tag <- if (length(args) >= 1) args[1] else "baseline"
set.seed(1)
N <- 30000L
ess_ps <- function(dr, secs) as.numeric(coda::effectiveSize(dr)) / max(secs, 0.02)
rows <- list()
add <- function(case, method, es, wall, est) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, method = method,
                                          ess_per_sec = es, wall_sec = wall,
                                          estimate = est)
}

## Case 1: mtcars mpg ~ wt (conjugate Gaussian regression)
yv <- mtcars$mpg; xw <- mtcars$wt; nobs <- length(yv)
m_wt <- gpum_model(~ -0.5 * ((mpg - a - b * wt) / exp(ls))^2 - ls,
                   params = c("a", "b", "ls"), data = c("mpg", "wt"))
dat1 <- list(mpg = yv, wt = xw)
for (meth in c("de", "rwm")) {
  t <- system.time(f <- gpu_metropolis(m_wt, data = dat1, n_iter = N,
        n_chains = 8, method = meth, warmup = if (meth == "rwm") "auto" else NULL, seed = 1, backend = "cpu"))["elapsed"]
  add("mtcars", paste0("gpum_", meth), ess_ps(as.vector(f$draws[, , "b"]), t),
      unname(t), mean(f$draws[, , "b"]))
}
if (exists("gpum_lm")) {
  t <- system.time(f <- gpu_metropolis(gpum_lm(mpg ~ wt, data = mtcars),
        n_iter = N, n_chains = 8, seed = 1))["elapsed"]
  add("mtcars", "gpum_exact", ess_ps(as.vector(f$draws[, , "wt"]), t),
      unname(t), mean(f$draws[, , "wt"]))
}
if (requireNamespace("mcmc", quietly = TRUE)) {
  lp <- function(th) sum(dnorm(yv, th[1] + th[2] * xw, exp(th[3]), log = TRUE)) -
          0.5 * (th[1] / 50)^2 - 0.5 * (th[2] / 50)^2
  t <- system.time(o <- mcmc::metrop(lp, c(20, -5, 1), nbatch = N,
                                     scale = c(2, 0.6, 0.1)))["elapsed"]
  add("mtcars", "mcmc", ess_ps(o$batch[, 2], t), unname(t), mean(o$batch[, 2]))
}
if (requireNamespace("MCMCpack", quietly = TRUE)) {
  t <- system.time(f <- MCMCpack::MCMCregress(mpg ~ wt, data = mtcars,
                                              mcmc = N, verbose = 0))["elapsed"]
  add("mtcars", "MCMCpack", ess_ps(f[, "wt"], t), unname(t), mean(f[, "wt"]))
}
if (requireNamespace("cmdstanr", quietly = TRUE)) {
  sc <- "data{int n; vector[n] y; vector[n] x;} parameters{real a; real b; real<lower=0> sigma;} model{ y ~ normal(a + b*x, sigma); a ~ normal(0,50); b ~ normal(0,50);}"
  mod <- cmdstanr::cmdstan_model(cmdstanr::write_stan_file(sc))
  t <- system.time(sf <- mod$sample(data = list(n = nobs, y = yv, x = xw),
        chains = 1, iter_sampling = N, iter_warmup = 1000, refresh = 0,
        show_messages = FALSE))["elapsed"]
  dr <- as.numeric(sf$draws("b", format = "draws_matrix"))
  add("mtcars", "Stan", ess_ps(dr, t), unname(t), mean(dr))
}

## Case 2: faithful waiting ~ eruptions (conjugate Gaussian regression, n=272)
d <- faithful
reg <- gpum_model(~ -0.5 * ((waiting - a - b * eruptions) / exp(ls))^2 - ls,
                  params = c("a", "b", "ls"), data = c("waiting", "eruptions"))
dat2 <- list(waiting = d$waiting, eruptions = d$eruptions)
for (meth in c("de", "rwm")) {
  t <- system.time(f <- gpu_metropolis(reg, data = dat2, n_iter = N,
        n_chains = 8, method = meth, warmup = if (meth == "rwm") "auto" else NULL, seed = 1, backend = "cpu"))["elapsed"]
  add("faithful", paste0("gpum_", meth), ess_ps(as.vector(f$draws[, , "b"]), t),
      unname(t), mean(f$draws[, , "b"]))
}
if (exists("gpum_lm")) {
  t <- system.time(f <- gpu_metropolis(gpum_lm(waiting ~ eruptions, data = d),
        n_iter = N, n_chains = 8, seed = 1))["elapsed"]
  add("faithful", "gpum_exact", ess_ps(as.vector(f$draws[, , "eruptions"]), t),
      unname(t), mean(f$draws[, , "eruptions"]))
}
if (requireNamespace("mcmc", quietly = TRUE)) {
  yv2 <- d$waiting; xv2 <- d$eruptions
  lp <- function(th) sum(dnorm(yv2, th[1] + th[2] * xv2, exp(th[3]), log = TRUE)) -
          0.5 * (th[1] / 80)^2 - 0.5 * (th[2] / 50)^2
  t <- system.time(o <- mcmc::metrop(lp, c(34, 11, 1.8), nbatch = N,
                                     scale = c(2, 0.6, 0.05)))["elapsed"]
  add("faithful", "mcmc", ess_ps(o$batch[, 2], t), unname(t), mean(o$batch[, 2]))
}
if (requireNamespace("MCMCpack", quietly = TRUE)) {
  t <- system.time(f <- MCMCpack::MCMCregress(waiting ~ eruptions, data = d,
                                              mcmc = N, verbose = 0))["elapsed"]
  add("faithful", "MCMCpack", ess_ps(f[, "eruptions"], t), unname(t),
      mean(f[, "eruptions"]))
}
if (requireNamespace("cmdstanr", quietly = TRUE)) {
  sc <- "data{int n; vector[n] y; vector[n] x;} parameters{real a; real b; real<lower=0> sigma;} model{ y ~ normal(a + b*x, sigma); a ~ normal(0,80); b ~ normal(0,50);}"
  mod <- cmdstanr::cmdstan_model(cmdstanr::write_stan_file(sc))
  t <- system.time(sf <- mod$sample(data = list(n = nrow(d), y = d$waiting,
        x = d$eruptions), chains = 1, iter_sampling = N, iter_warmup = 1000,
        refresh = 0, show_messages = FALSE))["elapsed"]
  dr <- as.numeric(sf$draws("b", format = "draws_matrix"))
  add("faithful", "Stan", ess_ps(dr, t), unname(t), mean(dr))
}

## Case 3: portpirie Gumbel (non-conjugate)
if (requireNamespace("evd", quietly = TRUE)) {
  y <- as.numeric(evd::portpirie)
  gum <- gpum_model(~ -(y - mu) / exp(lb) - exp(-(y - mu) / exp(lb)) - lb,
                    params = c("mu", "lb"), data = "y")
  dat3 <- list(y = y)
  for (meth in c("de", "rwm")) {
    t <- system.time(f <- gpu_metropolis(gum, data = dat3, n_iter = N,
          n_chains = 8, proposal_sd = c(0.04, 0.1), method = meth,
          warmup = if (meth == "rwm") "auto" else NULL,
          seed = 1, backend = "cpu"))["elapsed"]
    add("portpirie", paste0("gpum_", meth),
        ess_ps(as.vector(f$draws[, , "mu"]), t), unname(t),
        mean(f$draws[, , "mu"]))
  }
  if (requireNamespace("mcmc", quietly = TRUE)) {
    gll <- function(mu, lb) { z <- (y - mu) / exp(lb); sum(-z - exp(-z) - lb) }
    lp <- function(th) gll(th[1], th[2]) - 0.5 * ((th[1] - 4) / 2)^2 - 0.5 * (th[2] / 2)^2
    t <- system.time(o <- mcmc::metrop(lp, c(median(y), log(sd(y))),
                                       nbatch = N, scale = c(0.04, 0.1)))["elapsed"]
    add("portpirie", "mcmc", ess_ps(o$batch[, 1], t), unname(t), mean(o$batch[, 1]))
  }
  if (requireNamespace("cmdstanr", quietly = TRUE)) {
    sc <- "data{int n; vector[n] y;} parameters{real mu; real<lower=0> beta;} model{ y ~ gumbel(mu, beta); mu ~ normal(4,2); beta ~ normal(0,2);}"
    mod <- cmdstanr::cmdstan_model(cmdstanr::write_stan_file(sc))
    t <- system.time(sf <- mod$sample(data = list(n = length(y), y = y),
          chains = 1, iter_sampling = N, iter_warmup = 1000, refresh = 0,
          show_messages = FALSE))["elapsed"]
    dr <- as.numeric(sf$draws("mu", format = "draws_matrix"))
    add("portpirie", "Stan", ess_ps(dr, t), unname(t), mean(dr))
  }
}

res <- do.call(rbind, rows)
res$tag <- tag
out <- file.path("benchmark", paste0("opt_results_", tag, ".csv"))
write.csv(res, out, row.names = FALSE)
print(res[order(res$case, -res$ess_per_sec), ], row.names = FALSE)
cat("\nsaved:", out, "\n")
