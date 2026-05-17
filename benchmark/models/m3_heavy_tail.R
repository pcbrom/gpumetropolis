# Target model M3: a heavy-tailed location model, the tail-stability test of
# EXPERIMENT_PROTOCOL.md section 5.
#
# Observations y_1..y_N follow a Student t with 3 degrees of freedom, unit
# scale, unknown location mu, flat prior on mu. Three degrees of freedom give
# an undefined kurtosis and frequent outliers; the log-density has a soft,
# logarithmic penalty rather than the quadratic penalty of a Gaussian. The
# posterior of mu has no closed form, so the reference truth is built by
# one-dimensional quadrature of the log-posterior on a fine grid.

m3_spec <- local({
  true_mu <- 5
  nu <- 3
  scale <- 1

  make_data <- function(N, replication) {
    n_index <- match(N, c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7))
    if (is.na(n_index)) n_index <- 0L
    if (exists(".Random.seed", envir = .GlobalEnv)) {
      old <- get(".Random.seed", envir = .GlobalEnv)
      on.exit(assign(".Random.seed", old, envir = .GlobalEnv), add = TRUE)
    }
    set.seed(20260300L + 1000L * n_index + as.integer(replication))
    true_mu + scale * stats::rt(N, df = nu)
  }

  # Log posterior up to an additive constant. With nu = 3 the exponent
  # (nu + 1) / 2 equals 2.
  log_post <- function(theta, data) {
    sum(-0.5 * (nu + 1) * log1p((data - theta[1])^2 / (nu * scale^2)))
  }

  # Independent draws from the reference posterior, built by quadrature. The
  # posterior is unimodal; the grid is centred on its mode and spans many
  # asymptotic standard deviations, then the normalised mass yields an
  # inverse-CDF sampler.
  ref_sample <- function(data, n) {
    md <- stats::median(data)
    # Asymptotic posterior sd: inverse Fisher information of a t location.
    fisher <- (nu + 1) / ((nu + 3) * scale^2)
    approx_sd <- 1 / sqrt(length(data) * fisher)
    mode <- stats::optimize(function(m) log_post(m, data),
                            interval = md + c(-20, 20) * approx_sd,
                            maximum = TRUE)$maximum
    grid <- seq(mode - 18 * approx_sd, mode + 18 * approx_sd,
                length.out = 8001L)
    lp <- vapply(grid, function(m) log_post(m, data), numeric(1))
    w <- exp(lp - max(lp))
    cdf <- cumsum(w) / sum(w)
    matrix(stats::approx(cdf, grid, xout = stats::runif(n),
                         rule = 2L)$y, ncol = 1L)
  }

  init <- function(data, n_chains) {
    md <- stats::median(data)
    if (n_chains == 1L) {
      matrix(md, 1L, 1L)
    } else {
      matrix(md + scale * seq(-3, 3, length.out = n_chains), ncol = 1L)
    }
  }

  proposal_sd <- function(data) {
    fisher <- (nu + 1) / ((nu + 3) * scale^2)
    2.4 / sqrt(length(data) * fisher)
  }

  gpum_loglik <- stats::as.formula(
    sprintf("~ %.10g * log(1 + (y - mu)^2 / %.10g)",
            -0.5 * (nu + 1), nu * scale^2)
  )

  nimble_code <- quote({
    for (i in 1:N) {
      y[i] ~ dt(mu, tau, df)
    }
    mu ~ dflat()
  })
  nimble_constants <- function(data) {
    list(N = length(data), tau = 1 / scale^2, df = nu)
  }
  nimble_data <- function(data) list(y = data)
  nimble_inits <- function(data, n_chains) {
    starts <- as.numeric(init(data, n_chains))
    lapply(starts, function(m) list(mu = m))
  }

  stan_data <- function(data) {
    list(N = length(data), y = data, nu = nu, scale = scale)
  }

  list(
    id = "M3",
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
    stan_file = "benchmark/models/m3.stan",
    stan_data = stan_data,
    stan_pars = "mu"
  )
})
