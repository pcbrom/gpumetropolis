## Submission

This is a new submission.

## Test environments

- local: Ubuntu 24.04, R 4.6.0, rustc 1.95.0
- (to be completed: win-builder, R-hub, GitHub Actions)

## R CMD check results

0 errors | 0 warnings | notes to be confirmed on the CRAN-configuration build.

## Notes for the CRAN team

- The package compiles a Rust component through `cargo` (`SystemRequirements`
  lists `Cargo` and `rustc`). The Rust crates are vendored in the tarball, so
  the build does not need network access.
- The CUDA and Vulkan GPU backends are optional Cargo features, off by
  default. The default build is CPU only and needs no GPU toolchain. The GPU
  backends are documented for users who build the package with the
  corresponding feature and have the toolchain installed.
