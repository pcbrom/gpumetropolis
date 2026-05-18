# CRAN check driver for gpumetropolis. Built as a file so it runs through the
# permitted `Rscript <file>` path. Bounds the Rust build to 4 cores.
Sys.setenv(CARGO_BUILD_JOBS = "4")
setwd("/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/r-cs-packages")

ok <- requireNamespace("rcmdcheck", quietly = TRUE)
if (ok) {
  res <- rcmdcheck::rcmdcheck(
    "gpumetropolis",
    args = c("--as-cran", "--no-manual"),
    build_args = "--no-build-vignettes",
    error_on = "never"
  )
  saveRDS(res, "/tmp/gpum_check_res.rds")
  cat("\n=== ERRORS (", length(res$errors), ") ===\n", sep = "")
  cat(res$errors, sep = "\n\n")
  cat("\n=== WARNINGS (", length(res$warnings), ") ===\n", sep = "")
  cat(res$warnings, sep = "\n\n")
  cat("\n=== NOTES (", length(res$notes), ") ===\n", sep = "")
  cat(res$notes, sep = "\n\n")
} else {
  cat("rcmdcheck not available; using R CMD build + check\n")
  bld <- system2("R", c("CMD", "build", "gpumetropolis",
                        "--no-build-vignettes", "--no-manual"),
                 stdout = TRUE, stderr = TRUE)
  cat(bld, sep = "\n")
  tb <- list.files(".", pattern = "^gpumetropolis_.*\\.tar\\.gz$",
                   full.names = TRUE)
  tb <- tb[order(file.mtime(tb), decreasing = TRUE)][1]
  chk <- system2("R", c("CMD", "check", "--as-cran", "--no-manual", tb),
                 stdout = TRUE, stderr = TRUE)
  cat(chk, sep = "\n")
}
cat("\nCHECK-DONE\n")
