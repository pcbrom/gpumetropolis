# The version check on attach is silent on any failure and active only in
# interactive sessions, so the integration path cannot be tested directly
# from R CMD check. The JSON parser is testable in isolation and is the only
# stateful piece.

test_that(".gpum_parse_version extracts the Version field", {
  txt <- '{"package":"gpumetropolis","Version":"0.1.5","other":"x"}'
  expect_equal(gpumetropolis:::.gpum_parse_version(txt), "0.1.5")
})

test_that(".gpum_parse_version is tolerant of whitespace and line breaks", {
  txt <- '{\n  "Version"  :  "1.2.3"\n}'
  expect_equal(gpumetropolis:::.gpum_parse_version(txt), "1.2.3")
})

test_that(".gpum_parse_version returns NULL when Version is absent", {
  expect_null(gpumetropolis:::.gpum_parse_version('{"foo":"bar"}'))
})

test_that(".gpum_parse_version returns NULL on malformed payload", {
  expect_null(gpumetropolis:::.gpum_parse_version("not json at all"))
})
