# Correctness across distribution families: bimodal, heavy-tailed and
# multi-parameter models, declared in the DSL. These exercise the compiler,
# the JIT and the sampler on expressions beyond the Gaussian mean.

test_that("the sampler covers a bimodal posterior with many chains", {
  # y ~ Normal(|mu|, 1): the likelihood is symmetric in mu, so the posterior
  # of mu is bimodal at +3 and -3. A single random-walk chain stays in one
  # mode; many chains from spread starts cover both.
  set.seed(1)
  y <- rnorm(400, mean = 3, sd = 1)
  m <- gpum_model(
    loglik = ~ log(exp(-((y - mu)^2) / 2) + exp(-((y + mu)^2) / 2)),
    params = "mu", data = "y"
  )
  init <- matrix(seq(-6, 6, length.out = 16), nrow = 16, ncol = 1)
  fit <- gpu_metropolis(m, data = list(y = y), init = init,
                        proposal_sd = 0.15, n_iter = 4000, seed = 1,
                        backend = "cpu")
  pooled <- as.vector(fit$draws[2001:4000, , 1])
  # Both modes are represented in the pooled draws.
  expect_true(any(pooled > 1.5))
  expect_true(any(pooled < -1.5))
  # The modes sit near the expected magnitude.
  expect_equal(mean(pooled[pooled > 0]), 3, tolerance = 0.2)
  expect_equal(mean(pooled[pooled < 0]), -3, tolerance = 0.2)
})

test_that("the sampler recovers the location of a heavy-tailed model", {
  # Student t likelihood with 3 degrees of freedom: heavy tails, outliers.
  set.seed(2)
  y <- 5 + rt(800, df = 3)
  m <- gpum_model(
    loglik = ~ -2 * log(1 + (y - mu)^2 / 3),
    params = "mu", data = "y"
  )
  fit <- gpu_metropolis(m, data = list(y = y), proposal_sd = 0.08,
                        n_iter = 4000, n_chains = 6, seed = 2,
                        backend = "cpu")
  post <- fit$draws[2001:4000, , 1]
  expect_equal(mean(post), 5, tolerance = 0.25)
  expect_lt(rhat(fit$draws[, , 1]), 1.05)
})

test_that("the sampler recovers a two-parameter model", {
  # Normal with unknown mean mu and unknown log standard deviation ls.
  set.seed(3)
  y <- rnorm(3000, mean = 2, sd = 1.5)
  m <- gpum_model(
    loglik = ~ -((y - mu)^2) / (2 * exp(2 * ls)) - ls,
    params = c("mu", "ls"), data = "y"
  )
  init <- cbind(seq(-1, 5, length.out = 6), seq(-1, 1, length.out = 6))
  fit <- gpu_metropolis(m, data = list(y = y), init = init,
                        proposal_sd = c(0.03, 0.02), n_iter = 5000,
                        seed = 3, backend = "cpu")
  expect_equal(dim(fit$draws), c(5000L, 6L, 2L))
  mu_post <- fit$draws[2501:5000, , "mu"]
  ls_post <- fit$draws[2501:5000, , "ls"]
  expect_equal(mean(mu_post), 2, tolerance = 0.1)
  expect_equal(mean(ls_post), log(1.5), tolerance = 0.1)
})
