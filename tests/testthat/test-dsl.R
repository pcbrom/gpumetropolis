# Tests of the expression compiler: bytecode emission and validation.

test_that("the compiler emits correct bytecode for a known expression", {
  prog <- gpumetropolis:::.gpum_compile(
    gpumetropolis:::.gpum_rhs(~ mu + 2),
    params = "mu", data = character(0)
  )
  # PUSH_PARAM 0; PUSH_CONST 0; ADD
  expect_equal(prog$code, c(1L, 0L, 0L, 0L, 3L, 0L))
  expect_equal(prog$consts, 2)
})

test_that("the constant pool deduplicates repeated literals", {
  prog <- gpumetropolis:::.gpum_compile(
    gpumetropolis:::.gpum_rhs(~ 2 * mu + 2),
    params = "mu", data = character(0)
  )
  expect_equal(prog$consts, 2)
})

test_that("the compiler rejects unknown symbols and functions", {
  expect_error(gpumetropolis:::.gpum_compile(
    gpumetropolis:::.gpum_rhs(~ foo + mu), params = "mu", data = "y"))
  expect_error(gpumetropolis:::.gpum_compile(
    gpumetropolis:::.gpum_rhs(~ cos(mu)), params = "mu", data = "y"))
})

test_that("the compiler tracks operand-stack depth", {
  prog <- gpumetropolis:::.gpum_compile(
    gpumetropolis:::.gpum_rhs(~ (y - mu) * (y - mu)),
    params = "mu", data = "y"
  )
  expect_true(prog$depth >= 2L)
})
