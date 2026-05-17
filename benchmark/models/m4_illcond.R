# Target model M4: an ill-conditioned multivariate Gaussian, the geometric
# robustness test of EXPERIMENT_PROTOCOL.md section 5.
#
# The target is a three-dimensional Gaussian N(0, Sigma) with an
# equicorrelation Sigma at rho = 0.97. Its covariance has eigenvalues 2.94 and
# 0.03 (multiplicity two), a condition number of 98. The narrow direction is a
# rotation of the coordinate axes, so a per-coordinate proposal scale cannot
# remove the ill-conditioning; the random walk must take small steps to keep a
# workable acceptance rate, and mixing along the wide directions is slow. The
# reference truth is the closed-form Gaussian.
#
# Recorded scope note: M4 is a target with no observed data, so the data-size
# axis N of the factorial does not apply. M4 is therefore swept over the chain
# count C only, at a single nominal N; the loglik carries a 0 * y term so the
# declaration still names a data column, and the data is the single value 0.

m4_spec <- local({
  rho <- 0.97
  d <- 3L
  R <- matrix(rho, d, d)
  diag(R) <- 1
  Sigma <- R
  P <- solve(Sigma)
  c1 <- -0.5 * P[1, 1]    # coefficient on t_k^2, equal on the diagonal
  c2 <- -P[1, 2]          # coefficient on t_j t_k, equal off the diagonal

  # No observed data; the data column exists only to satisfy the DSL. N is
  # ignored. The value is fixed so every replication and backend agree.
  make_data <- function(N, replication) 0

  # Log posterior up to an additive constant; theta is the length-three
  # parameter vector. The data argument is unused.
  log_post <- function(theta, data) {
    theta <- as.numeric(theta)
    -0.5 * as.numeric(crossprod(theta, P %*% theta))
  }

  # Independent draws from the reference truth, the closed-form Gaussian.
  ref_sample <- function(data, n) {
    L <- chol(Sigma)
    matrix(stats::rnorm(n * d), n, d) %*% L
  }

  # Overdispersed starts: a spread along the wide direction, staggered across
  # coordinates so the narrow direction is exercised too.
  init <- function(data, n_chains) {
    stagger <- c(-1.5, 0, 1.5)
    if (n_chains == 1L) {
      matrix(stagger, 1L, d)
    } else {
      s <- seq(-3, 3, length.out = n_chains)
      t(vapply(s, function(v) v + stagger, numeric(d)))
    }
  }

  # Per-coordinate proposal scaled to the narrow direction, so the acceptance
  # rate stays workable; the same scale is given to every backend.
  proposal_sd <- function(data) {
    rep(2.4 / sqrt(d) * sqrt(1 - rho), d)
  }

  # gpumetropolis DSL: the quadratic form written out, plus a 0 * y term so
  # the declaration names the data column y.
  gpum_loglik <- stats::as.formula(sprintf(
    paste0("~ %.10g * (t1^2 + t2^2 + t3^2) + %.10g * ",
           "(t1*t2 + t1*t3 + t2*t3) + 0 * y"),
    c1, c2
  ))

  nimble_code <- quote({
    theta[1:3] ~ dmnorm(zeros[1:3], prec = Prec[1:3, 1:3])
  })
  nimble_constants <- function(data) {
    list(zeros = rep(0, d), Prec = P)
  }
  nimble_data <- function(data) list()
  nimble_inits <- function(data, n_chains) {
    starts <- init(data, n_chains)
    lapply(seq_len(n_chains), function(k) list(theta = starts[k, ]))
  }

  stan_data <- function(data) list(P = P)

  list(
    id = "M4",
    params = c("t1", "t2", "t3"),
    data_names = "y",
    dim = d,
    sweeps_n = FALSE,
    Sigma = Sigma,
    P = P,
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
    nimble_monitors = "theta",
    stan_file = "benchmark/models/m4.stan",
    stan_data = stan_data,
    stan_pars = "theta"
  )
})
