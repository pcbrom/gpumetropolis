//! GPU-portable log-density kernels via CubeCL.
//!
//! One kernel source compiles to the CPU runtime and to GPU runtimes. This
//! module currently ships the batched Gaussian-mean log-density; the runtime
//! expression compiler is built on top of the same dispatch path.
//!
//! Parallelism: one thread per chain. Each thread evaluates the log-density of
//! its chain's candidate by looping over the data. The chain axis, which the
//! scope analysis identified as the dominant one, is the parallel axis here.

use cubecl::prelude::*;

/// Batched Gaussian-mean log-density.
///
/// `candidates` holds one mean per chain; `out` receives one log-density per
/// chain. `inv_two_var` is passed as a one-element array, which keeps the
/// launch on the array-argument path.
#[cube(launch_unchecked)]
fn gaussian_logdens_kernel(
    candidates: &Array<f32>,
    data: &Array<f32>,
    inv_two_var: &Array<f32>,
    out: &mut Array<f32>,
) {
    if ABSOLUTE_POS < candidates.len() {
        let mu = candidates[ABSOLUTE_POS];
        let mut ss = 0.0f32;
        for i in 0..data.len() {
            let d = data[i] - mu;
            ss += d * d;
        }
        out[ABSOLUTE_POS] = -inv_two_var[0] * ss;
    }
}

/// Compute backend for the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
}

impl Backend {
    /// Parse a backend name; unknown names fall back to the CPU.
    pub fn from_name(name: &str) -> Backend {
        match name.to_ascii_lowercase().as_str() {
            "cuda" | "gpu" => Backend::Cuda,
            _ => Backend::Cpu,
        }
    }
}

fn run<R: Runtime>(candidates: &[f32], data: &[f32], inv_two_var: f32) -> Vec<f32> {
    let client = R::client(&Default::default());
    let n_chains = candidates.len();

    let cand_h = client.create_from_slice(f32::as_bytes(candidates));
    let data_h = client.create_from_slice(f32::as_bytes(data));
    let itv = [inv_two_var];
    let itv_h = client.create_from_slice(f32::as_bytes(&itv));
    let out_h = client.empty(n_chains * core::mem::size_of::<f32>());

    let block = 256u32;
    let count = CubeCount::Static((n_chains as u32).div_ceil(block), 1, 1);
    let dim = CubeDim { x: block, y: 1, z: 1 };

    unsafe {
        gaussian_logdens_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(cand_h, n_chains),
            ArrayArg::from_raw_parts(data_h, data.len()),
            ArrayArg::from_raw_parts(itv_h, 1),
            ArrayArg::from_raw_parts(out_h.clone(), n_chains),
        );
    }

    let bytes = client.read_one_unchecked(out_h);
    f32::from_bytes(&bytes).to_vec()
}

/// Evaluate the batched Gaussian-mean log-density on the chosen backend.
pub fn gaussian_logdens(
    candidates: &[f32],
    data: &[f32],
    sigma: f32,
    backend: Backend,
) -> Vec<f32> {
    let inv_two_var = 0.5 / (sigma * sigma);
    match backend {
        Backend::Cpu => run::<cubecl::cpu::CpuRuntime>(candidates, data, inv_two_var),
        Backend::Cuda => run::<cubecl::cuda::CudaRuntime>(candidates, data, inv_two_var),
    }
}
