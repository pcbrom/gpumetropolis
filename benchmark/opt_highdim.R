# v0.5.2 high-dimension head-to-head: Bayesian logistic regression with 20
# covariates plus intercept (d = 21), n = 500, simulated with a fixed seed.
# This is the regime the 0.5.0 verdict left to the gradient samplers: the
# random-walk step must shrink as d^{-1/2} while the Langevin step shrinks
# as d^{-1/6}, so MALA is the package's answer here. Stan (NUTS) is the
# reference competitor; gpum_rwm is the internal baseline that shows what
# the gradient buys.
#
# Usage: Rscript benchmark/opt_highdim.R <tag>
# Writes benchmark/opt_highdim_<tag>.csv

suppressMessages({library(gpumetropolis); library(coda)})
args <- commandArgs(trailingOnly = TRUE)
tag <- if (length(args) >= 1) args[1] else "r1"
N <- 30000L
D <- 20L
NOBS <- 500L
ess_ps <- function(dr, secs) as.numeric(coda::effectiveSize(dr)) / max(secs, 0.02)
rows <- list()
add <- function(method, es, wall, est) {
  rows[[length(rows) + 1]] <<- data.frame(case = "logistic_d21",
                                          method = method, ess_per_sec = es,
                                          wall_sec = wall, estimate = est)
}

set.seed(42)
X <- matrix(rnorm(NOBS * D), NOBS, D)
beta_true <- c(0.8, -0.6, rep(c(0.4, -0.3, 0.2, 0), 5)[seq_len(D - 2)])
eta_true <- -0.2 + X %*% beta_true
yb <- rbinom(NOBS, 1, 1 / (1 + exp(-eta_true)))

## gpumetropolis: DSL formula built programmatically
covs <- paste0("x", seq_len(D))
pars <- c("b0", paste0("b", seq_len(D)))
eta <- paste("b0", paste(sprintf("b%d * x%d", seq_len(D), seq_len(D)),
                         collapse = " + "), sep = " + ")
ll <- stats::as.formula(sprintf("~ yb * (%s) - log(1 + exp(%s))", eta, eta))
pr <- stats::as.formula(sprintf("~ %s", paste(
  sprintf("-0.5 * (b%s / 5)^2", c(0, seq_len(D))), collapse = " + ")))
m <- gpum_model(ll, params = pars, data = c("yb", covs), prior = pr)
dat <- c(list(yb = yb), stats::setNames(lapply(seq_len(D), function(j) X[, j]), covs))

for (meth in c("mala", "rwm")) {
  t <- system.time(f <- gpu_metropolis(m, data = dat, n_iter = N,
        n_chains = 8, method = meth, warmup = "auto", seed = 1,
        backend = "cpu"))["elapsed"]
  add(paste0("gpum_", meth), ess_ps(as.vector(f$draws[, , "b1"]), t),
      unname(t), mean(f$draws[, , "b1"]))
}

## Stan (NUTS), 1 chain, its idiomatic configuration
if (requireNamespace("cmdstanr", quietly = TRUE)) {
  sc <- "data{int n; int d; matrix[n,d] X; array[n] int y;}
parameters{real b0; vector[d] b;}
model{ y ~ bernoulli_logit(b0 + X * b); b0 ~ normal(0,5); b ~ normal(0,5);}"
  mod <- cmdstanr::cmdstan_model(cmdstanr::write_stan_file(sc))
  t <- system.time(sf <- mod$sample(data = list(n = NOBS, d = D, X = X,
        y = as.integer(yb)), chains = 1, iter_sampling = N,
        iter_warmup = 1000, refresh = 0, show_messages = FALSE))["elapsed"]
  dr <- as.numeric(sf$draws("b[1]", format = "draws_matrix"))
  add("Stan", ess_ps(dr, t), unname(t), mean(dr))
}

res <- do.call(rbind, rows)
res$tag <- tag
out <- file.path("benchmark", paste0("opt_highdim_", tag, ".csv"))
write.csv(res, out, row.names = FALSE)
print(res[order(-res$ess_per_sec), ], row.names = FALSE)
cat("\nsaved:", out, "\n")
