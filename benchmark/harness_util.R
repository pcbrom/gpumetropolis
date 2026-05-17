# Shared harness utilities: ESS, thinning and the H1 correctness gate.
# Used by run_one.R and by the analysis scripts so the definitions are single
# sourced.

suppressPackageStartupMessages(library(coda))

# Total effective sample size, summed over chains, with coda as the single
# uniform estimator (protocol section 6.2).
total_ess <- function(draws) {
  chains <- lapply(seq_len(ncol(draws)), function(ch) coda::mcmc(draws[, ch]))
  sum(coda::effectiveSize(coda::as.mcmc.list(chains)))
}

# Evenly spaced subsample down to k draws.
thin_to <- function(v, k) {
  k <- max(2L, min(as.integer(k), length(v)))
  if (k >= length(v)) return(v)
  v[round(seq(1, length(v), length.out = k))]
}

# H1 correctness gate, per replication. Records the two-sample KS test of the
# thinned pooled draws against an exact sample from the closed-form posterior,
# and split R-hat. The H1 verdict itself is taken downstream, on the family of
# all KS p-values, with Holm-Bonferroni control (protocol section 6.1, v0.2).
h1_gate <- function(draws, truth) {
  pooled <- as.vector(draws)
  thinned <- thin_to(pooled, floor(total_ess(draws)))
  reference <- stats::rnorm(length(thinned), truth$mean, truth$sd)
  ks <- suppressWarnings(stats::ks.test(thinned, reference))
  rh <- tryCatch(gpumetropolis::rhat(draws, warmup = 0L),
                 error = function(e) NA_real_)
  list(ks_stat = unname(ks$statistic), ks_pvalue = ks$p.value, rhat = rh)
}

# H1 correctness gate for the M2 to M4 factorial, per replication. `draws` is
# the (n_keep, n_chains, dim) array returned by the v2 adapters; `ref_fn(n)`
# returns an (n, dim) matrix of fresh independent draws from the reference
# truth. Each coordinate's thinned pooled draws are KS-tested against the
# matching reference column. The replication-level p-value is the smallest
# coordinate p-value with a Bonferroni adjustment by `dim`, so a single honest
# p-value enters the downstream Holm-Bonferroni family; the recorded statistic
# and R-hat are the worst over coordinates.
h1_gate_ms <- function(draws, ref_fn) {
  dm <- dim(draws)
  d <- dm[3]
  pv <- numeric(d)
  st <- numeric(d)
  rh <- numeric(d)
  for (j in seq_len(d)) {
    dj <- matrix(draws[, , j], dm[1], dm[2])
    pooled <- as.vector(dj)
    thinned <- thin_to(pooled, floor(total_ess(dj)))
    refj <- ref_fn(length(thinned))[, j]
    ks <- suppressWarnings(stats::ks.test(thinned, refj))
    pv[j] <- ks$p.value
    st[j] <- unname(ks$statistic)
    rh[j] <- tryCatch(gpumetropolis::rhat(dj, warmup = 0L),
                      error = function(e) NA_real_)
  }
  list(ks_stat = max(st),
       ks_pvalue = min(1, d * min(pv)),
       rhat = max(rh, na.rm = TRUE))
}

# Mean per-coordinate total ESS: the throughput numerator for a model of any
# dimension. Within a model the coordinate count is constant, so this keeps
# the ESS/s comparison across backends unbiased and commensurable with the
# one-dimensional models.
mean_total_ess <- function(draws) {
  dm <- dim(draws)
  mean(vapply(seq_len(dm[3]), function(j) {
    total_ess(matrix(draws[, , j], dm[1], dm[2]))
  }, numeric(1)))
}
