# CubeCL integration proof of concept

De-risking artefact for Phase 1 of the canonical plan. Confirms that one CubeCL
kernel source compiles and runs on the CPU runtime and the CUDA runtime, before
the expression compiler is built on top of CubeCL.

The kernel is the per-observation squared deviation `(x_i - mu)^2`, the
data-parallel core of the Gaussian log-density.

## Result (2026-05-16)

```
CPU : n=4096 sum=1970731.3750 max_err=0.00e0
CUDA: n=4096 sum=1970731.3750 max_err=0.00e0
```

Both backends, driven from the same `#[cube]` source, match a plain Rust
reference to the last bit on this input. CubeCL 0.10.0, host as recorded in
`benchmark/ENVIRONMENT.md` (RTX 4090, nvcc 12.0).

## What this de-risks, and what it does not

De-risked: CubeCL builds on the host; one kernel source targets CPU and CUDA;
the vendor-agnostic compile path is real.

Still open: integrating CubeCL into the extendr build of the R package; the AMD
and WGPU/Vulkan backends; and that the CubeCL dependency tree is large (about
414 crates), which will weigh on the vendored CRAN tarball and is tracked as a
Phase 3/4 risk.

## Run

```bash
cargo run --release
```
