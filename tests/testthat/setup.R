# Cap the Rayon thread pool of the native CPU backend at two threads for the
# whole test run. The CRAN check farm allows at most two cores; the test
# suite does not need more. Rayon reads RAYON_NUM_THREADS the first time its
# pool is built, and this setup file runs before any test, so the cap takes
# effect for every backend call in the suite.
Sys.setenv(RAYON_NUM_THREADS = "2")
