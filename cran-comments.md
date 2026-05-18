## Submission

This is a new submission.

## Test environments

- local: Ubuntu 24.04, R 4.6.0, rustc 1.95.0
- (to be completed: win-builder, R-hub, GitHub Actions)

## R CMD check results

Local check, `R CMD build` then `R CMD check --as-cran` with the vignette
built: 0 errors | 1 warning | 2 notes. None is a defect of the package; each
is an artifact of the local environment or of a not-yet-published commit.

- WARNING, checking top-level files: "A complete check needs the
  'checkbashisms' script." That script is not installed in the local check
  environment; it is present on the CRAN check machines, where the warning
  does not arise. The package's only shell script, `configure`, is plain
  POSIX sh.
- NOTE, checking compilation flags used: the local R was built with
  `-mno-omit-leaf-frame-pointer` in its default CFLAGS, so the check reports
  the flag for every compiled package. It is a property of the local R, not
  of this package, and does not arise on the CRAN configuration.
- NOTE, checking CRAN incoming feasibility: "New submission", as expected;
  and two README URLs to per-cell benchmark summary files reported 404 at
  check time because the commits that add those files were not yet pushed to
  the public repository. They resolve once the repository is updated.

The win-builder and R-hub results, on environments without the two local
artifacts above, will be added before submission.

## Notes for the CRAN team

- The package compiles a Rust component through `cargo` (`SystemRequirements`
  lists `Cargo` and `rustc`). The Rust crates are vendored in the tarball, so
  the build does not need network access.
- The installed size of the tarball is dominated by `src/rust/vendor.tar.xz`,
  the compressed vendored Rust sources. The kernel that runs on the CPU and on
  the GPU from one source is built with CubeCL, whose dependency tree is large.
  CRAN policy accepts vendored Rust sources as the reason for a larger tarball;
  the archive is already `xz`-compressed at the maximum setting. The vendored
  sources are build-time only and are removed after compilation, so they do
  not contribute to the installed package size.
- The CUDA and Vulkan GPU backends are optional Cargo features, off by
  default. The default build is CPU only and needs no GPU toolchain. The GPU
  backends are documented for users who build the package with the
  corresponding feature and have the toolchain installed.
