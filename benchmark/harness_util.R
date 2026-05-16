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
