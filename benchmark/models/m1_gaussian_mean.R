# Target model M1: Gaussian mean with known sigma and a flat prior.
#
# Observations y_1..y_N ~ Normal(mu, sigma^2), sigma known. With a flat prior on
# mu the posterior is closed form, Normal(mean(y), sigma^2 / N), which is the
# exact reference truth used by the H1 correctness gate and the H5 numerical
# checks of EXPERIMENT_PROTOCOL.md.

m1_spec <- local({
  true_mu <- 0.37
  sigma <- 1.5

  # Data depends only on (N, replication), never on the backend, so every
  # sampler sees the same data set in a given replication. This is deliberate:
  # the seed scheme of the protocol fixes the sampler seed, while the data seed
  # is kept backend-independent to avoid confounding data with backend.
  make_data <- function(N, replication) {
    n_index <- match(N, c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7))
    if (is.na(n_index)) n_index <- 0L
    if (exists(".Random.seed", envir = .GlobalEnv)) {
      old <- get(".Random.seed", envir = .GlobalEnv)
      on.exit(assign(".Random.seed", old, envir = .GlobalEnv), add = TRUE)
    }
    set.seed(20260000L + 1000L * n_index + as.integer(replication))
    stats::rnorm(N, mean = true_mu, sd = sigma)
  }

  # Log posterior up to an additive constant. Metropolis needs the density only
  # up to a constant; the dropped term does not depend on mu.
  log_post <- function(mu, data) {
    -0.5 / sigma^2 * sum((data - mu)^2)
  }

  # Exact posterior of mu given the data.
  truth <- function(data) {
    list(mean = mean(data), sd = sigma / sqrt(length(data)))
  }

  list(
    id = "M1",
    true_mu = true_mu,
    sigma = sigma,
    dim = 1L,
    make_data = make_data,
    log_post = log_post,
    truth = truth
  )
})
