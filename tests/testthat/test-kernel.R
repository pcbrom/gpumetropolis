# Direct checks of the batched log-density kernel. The kernel is the swappable
# compute unit; later phases must keep these values bit-for-bit.

test_that("log-density kernel matches the closed form", {
  # Target proportional to -0.5 / sigma^2 * sum_i (data_i - mu)^2.
  ld <- gpumetropolis:::rust_log_density_batch(
    candidates = c(0, 1, 2),
    data = c(0, 0, 0),
    sigma = 1
  )
  # mu = 0: sum sq = 0; mu = 1: sum sq = 3; mu = 2: sum sq = 12.
  expect_equal(ld, c(0, -1.5, -6))
})

test_that("log-density kernel scales with sigma", {
  ld1 <- gpumetropolis:::rust_log_density_batch(2, c(0, 0), sigma = 1)
  ld2 <- gpumetropolis:::rust_log_density_batch(2, c(0, 0), sigma = 2)
  # Halving the inverse variance by a factor of four when sigma doubles.
  expect_equal(ld1, -4)
  expect_equal(ld2, -1)
})

test_that("log-density kernel returns one value per candidate", {
  ld <- gpumetropolis:::rust_log_density_batch(
    as.numeric(seq_len(7)), rnorm(10), sigma = 1
  )
  expect_length(ld, 7L)
})
