## Submission

This is a new submission.

## Test environments

- local: Ubuntu 24.04, R 4.6.0, rustc 1.95.0
- win-builder: R-devel and R-release (Windows)

## R CMD check results

win-builder, R-devel and R-release: 0 errors | 0 warnings | 1 note.

The note is the same on both: "Possibly misspelled words in DESCRIPTION:
Gelman, bytecode". Both are correct. "Gelman" is the surname of A. Gelman,
in the reference "Gelman and Rubin (1992)" cited in the Description for the
split R-hat statistic; "bytecode" is the standard computing term for the
compiled stack-machine representation the package uses. There is no other
note, warning or error on win-builder.

The local check additionally reports a "checkbashisms" warning and a
"compilation flags used" note; both are artifacts of the local machine, the
missing `checkbashisms` script and the local R's CFLAGS, and neither arises
on win-builder, as the result above shows.

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
