# Target model M2: a separated bimodal posterior, the mode-crossing test of
# EXPERIMENT_PROTOCOL.md section 5.
#
# Observations y_1..y_N ~ Normal(|mu|, sigma), sigma known, flat prior on mu.
# The likelihood depends on mu only through |mu|, so the posterior of mu is
# symmetric and bimodal, with modes near +c and -c where c = mean(|y|). A
# single random-walk chain settles in one basin; crossing the low-density
# region near mu = 0 is exponentially rare once the modes separate. Many
# chains from spread starts cover both. This is a property of random-walk
# Metropolis, identical for every backend, and the model is included to make
# that property measurable rather than to expose a backend defect.

m2_spec <- local({
  true_c <- 3
  sigma <- 1

  # Data is backend-independent and fixed by (N, replication), as in M1.
  make_data <- function(N, replication) {
    n_index <- match(N, c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7))
    if (is.na(n_index)) n_index <- 0L
    if (exists(".Random.seed", envir = .GlobalEnv)) {
      old <- get(".Random.seed", envir = .GlobalEnv)
      on.exit(assign(".Random.seed", old, envir = .GlobalEnv), add = TRUE)
    }
    set.seed(20260200L + 1000L * n_index + as.integer(replication))
    stats::rnorm(N, mean = true_c, sd = sigma)
  }

  # Log posterior up to an additive constant; theta is the length-one vector
  # holding mu.
  log_post <- function(theta, data) {
    -0.5 / sigma^2 * sum((data - abs(theta[1]))^2)
  }

  # Independent draws from the reference posterior. The posterior of m = |mu|
  # is, to a negligible truncation, Normal(mean(data), sigma^2 / N); the sign
  # of mu is plus or minus one with equal probability, independent of m. The
  # reference is therefore exact up to the mass of the truncated tail at
  # m <= 0, which is astronomically small in every swept cell.
  ref_sample <- function(data, n) {
    m_mean <- mean(data)
    m_sd <- sigma / sqrt(length(data))
    m <- abs(stats::rnorm(n, m_mean, m_sd))
    s <- sample(c(-1, 1), n, replace = TRUE)
    matrix(s * m, ncol = 1L)
  }

  # Overdispersed starts spanning both basins.
  init <- function(data, n_chains) {
    if (n_chains == 1L) {
      matrix(mean(data), 1L, 1L)
    } else {
      matrix(seq(-2 * true_c, 2 * true_c, length.out = n_chains), ncol = 1L)
    }
  }

  proposal_sd <- function(data) 2.4 * sigma / sqrt(length(data))

  # gpumetropolis DSL: |mu| is written sqrt(mu^2), inside the supported
  # operation set. sigma = 1, so the divisor 2 * sigma^2 is 2.
  gpum_loglik <- stats::as.formula(
    sprintf("~ -((y - sqrt(mu^2))^2) / %.10g", 2 * sigma^2)
  )

  # nimble: |mu| through abs(); the flat prior matches the other backends.
  nimble_code <- quote({
    for (i in 1:N) {
      y[i] ~ dnorm(absmu, sd = sigma)
    }
    absmu <- abs(mu)
    mu ~ dflat()
  })
  nimble_constants <- function(data) {
    list(N = length(data), sigma = sigma)
  }
  nimble_data <- function(data) list(y = data)
  nimble_inits <- function(data, n_chains) {
    starts <- as.numeric(init(data, n_chains))
    lapply(starts, function(m) list(mu = m))
  }

  stan_data <- function(data) {
    list(N = length(data), y = data, sigma = sigma)
  }

  list(
    id = "M2",
    params = "mu",
    data_names = "y",
    dim = 1L,
    sweeps_n = TRUE,
    make_data = make_data,
    log_post = log_post,
    ref_sample = ref_sample,
    init = init,
    proposal_sd = proposal_sd,
    gpum_loglik = gpum_loglik,
    nimble_code = nimble_code,
    nimble_constants = nimble_constants,
    nimble_data = nimble_data,
    nimble_inits = nimble_inits,
    nimble_monitors = "mu",
    stan_file = "benchmark/models/m2.stan",
    stan_data = stan_data,
    stan_pars = "mu"
  )
})
