// CubeCL integration proof of concept for gpumetropolis.
//
// Goal: confirm that one CubeCL kernel source compiles and runs on the CPU
// runtime and the CUDA runtime, before committing to the expression compiler.
// The kernel is the per-observation squared deviation, the data-parallel core
// of the Gaussian log-density.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn squared_dev_kernel(x: &Array<f32>, mu: &Array<f32>, out: &mut Array<f32>) {
    if ABSOLUTE_POS < x.len() {
        let d = x[ABSOLUTE_POS] - mu[0];
        out[ABSOLUTE_POS] = d * d;
    }
}

fn run<R: Runtime>(name: &str, device: &R::Device) {
    let client = R::client(device);

    let data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.01).collect();
    let mu = vec![2.0f32];
    let n = data.len();

    let x = client.create_from_slice(f32::as_bytes(&data));
    let mu_h = client.create_from_slice(f32::as_bytes(&mu));
    let out = client.empty(n * core::mem::size_of::<f32>());

    let block = 256u32;
    let dim = CubeDim { x: block, y: 1, z: 1 };
    let count = CubeCount::Static((n as u32).div_ceil(block), 1, 1);

    unsafe {
        squared_dev_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(x, n),
            ArrayArg::from_raw_parts(mu_h, 1),
            ArrayArg::from_raw_parts(out.clone(), n),
        );
    }

    let bytes = client.read_one_unchecked(out);
    let result = f32::from_bytes(&bytes);

    let expect: Vec<f32> = data.iter().map(|v| (v - mu[0]) * (v - mu[0])).collect();
    let max_err = result
        .iter()
        .zip(&expect)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let sum: f32 = result.iter().sum();
    println!("{name}: n={n} sum={sum:.4} max_err={max_err:.2e}");
}

fn main() {
    run::<cubecl::cpu::CpuRuntime>("CPU ", &Default::default());
    run::<cubecl::cuda::CudaRuntime>("CUDA", &Default::default());
}
