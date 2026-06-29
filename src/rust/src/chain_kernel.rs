//! Whole-chain Metropolis kernel, one block per chain.
//!
//! One CubeCL kernel runs the entire sampler. A block (workgroup) of `BLOCK`
//! threads is assigned to one chain. The `n_iter` Metropolis steps run
//! sequentially for the block; within each step the threads of the block split
//! the sum over the observations and combine their partial sums by a tree
//! reduction in shared memory. The data sum, the cost that dominates, is
//! `BLOCK`-way parallel.
//!
//! This serves both regimes: few chains with large data (the threads of the
//! one block parallelise the data sum) and many chains (one block each). The
//! time axis stays sequential within a block, consistent with the caveat that
//! a chain does not parallelise; the parallelism is the data sum and the
//! chains.

use cubecl::prelude::*;

const STACK: usize = 32;
const MAX_PARAMS: usize = 16;
const BLOCK: u32 = 256;

/// triple32 integer hash (Wellons). A high-quality `u32 -> u32` mix.
#[cube]
fn triple32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 17;
    h *= 2129725213u32;
    h ^= h >> 11;
    h *= 2890025505u32;
    h ^= h >> 15;
    h *= 831628273u32;
    h ^= h >> 14;
    h
}

/// One uniform in (0, 1) from the counter.
#[cube]
fn rand_uniform(seedmix: u32, ctr: u32) -> f32 {
    let h = triple32(seedmix + ctr);
    (f32::cast_from(h) + 0.5) / 4294967296.0
}

/// Run one bytecode program once and return the scalar result. A program with
/// no instructions returns zero.
#[cube]
fn vm_eval(
    code: &Array<u32>,
    consts: &Array<f32>,
    n_instr: usize,
    params: &Array<f32>,
    data: &Array<f32>,
    data_base: usize,
) -> f32 {
    let mut stack = Array::<f32>::new(STACK);
    let mut sp = 0usize;
    let mut pc = 0usize;
    while pc < n_instr {
        let op = code[2 * pc];
        let arg = usize::cast_from(code[2 * pc + 1]);
        if op == 0 {
            stack[sp] = consts[arg];
            sp += 1;
        } else if op == 1 {
            stack[sp] = params[arg];
            sp += 1;
        } else if op == 2 {
            stack[sp] = data[data_base + arg];
            sp += 1;
        } else if op == 3 {
            stack[sp - 2] = stack[sp - 2] + stack[sp - 1];
            sp -= 1;
        } else if op == 4 {
            stack[sp - 2] = stack[sp - 2] - stack[sp - 1];
            sp -= 1;
        } else if op == 5 {
            stack[sp - 2] = stack[sp - 2] * stack[sp - 1];
            sp -= 1;
        } else if op == 6 {
            stack[sp - 2] = stack[sp - 2] / stack[sp - 1];
            sp -= 1;
        } else if op == 7 {
            stack[sp - 1] = -stack[sp - 1];
        } else if op == 8 {
            stack[sp - 1] = stack[sp - 1].exp();
        } else if op == 9 {
            stack[sp - 1] = stack[sp - 1].ln();
        } else if op == 10 {
            stack[sp - 1] = stack[sp - 1].sqrt();
        } else {
            stack[sp - 2] = stack[sp - 2].powf(stack[sp - 1]);
            sp -= 1;
        }
        pc += 1;
    }
    let mut result = 0.0f32;
    if n_instr > 0 {
        result = stack[0];
    }
    result
}

/// One block per chain; `BLOCK` threads per block share the data sum.
#[cube(launch_unchecked)]
fn metropolis_block_kernel(
    code: &Array<u32>,
    consts: &Array<f32>,
    prior_code: &Array<u32>,
    prior_consts: &Array<f32>,
    data: &Array<f32>,
    init: &Array<f32>,
    proposal_sd: &Array<f32>,
    temperatures: &Array<f32>,
    de_params: &Array<f32>,   // [gamma, de_noise]
    meta: &Array<u32>,        // [n_instr, n_params, n_cols, n_obs, n_iter, prior_n, seed, proposal_mode]
    out_draws: &mut Array<f32>,
    out_accept: &mut Array<f32>,
    out_logpost: &mut Array<f32>,
) {
    let chain = usize::cast_from(CUBE_POS);
    let tid = usize::cast_from(UNIT_POS);
    let block = usize::cast_from(BLOCK);
    let n_chains = out_accept.len();

    let mut state = SharedMemory::<f32>::new(MAX_PARAMS);
    let mut prop = SharedMemory::<f32>::new(MAX_PARAMS);
    let mut partials = SharedMemory::<f32>::new(256usize);

    if chain < n_chains {
        let n_instr = usize::cast_from(meta[0]);
        let n_params = usize::cast_from(meta[1]);
        let n_cols = usize::cast_from(meta[2]);
        let n_obs = usize::cast_from(meta[3]);
        let n_iter = usize::cast_from(meta[4]);
        let prior_n = usize::cast_from(meta[5]);
        let seed = meta[6];
        let proposal_mode = meta[7];
        let pbase = chain * n_params;
        // The base seed is hashed before the chain offset is added, so two
        // runs with consecutive integer seeds get well-separated counter
        // streams rather than streams that overlap by a one-counter shift.
        let seedmix = triple32(seed) + u32::cast_from(chain) * 2654435761u32;

        // Thread 0 seeds the chain state from `init`.
        if tid == 0 {
            let mut j = 0usize;
            while j < n_params {
                state[j] = init[pbase + j];
                j += 1;
            }
        }
        sync_cube();

        // Initial log-likelihood: each thread sums its slice of observations,
        // then a tree reduction in shared memory combines the partials.
        let mut local = Array::<f32>::new(MAX_PARAMS);
        let mut j = 0usize;
        while j < n_params {
            local[j] = state[j];
            j += 1;
        }
        let mut part = 0.0f32;
        let mut obs = tid;
        while obs < n_obs {
            part += vm_eval(code, consts, n_instr, &local, data, obs * n_cols);
            obs += block;
        }
        partials[tid] = part;
        sync_cube();
        let mut stride = block / 2usize;
        while stride >= 1usize {
            if tid < stride {
                partials[tid] = partials[tid] + partials[tid + stride];
            }
            sync_cube();
            stride /= 2usize;
        }

        let mut cur = 0.0f32;
        let mut accepts: u32 = 0u32;
        let mut ctr: u32 = 0u32;
        if tid == 0 {
            cur = partials[0] + vm_eval(prior_code, prior_consts, prior_n,
                                       &local, data, 0);
        }

        let two_pi = 6.2831853f32;
        let mut t = 0usize;
        while t < n_iter {
            // Thread 0 proposes; the block sees the proposal after the barrier.
            if tid == 0 {
                if proposal_mode == 0u32 {
                    // Gaussian random walk.
                    let mut k = 0usize;
                    while k < n_params {
                        ctr += 1u32;
                        let u1 = rand_uniform(seedmix, ctr);
                        ctr += 1u32;
                        let u2 = rand_uniform(seedmix, ctr);
                        let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                        prop[k] = state[k] + proposal_sd[pbase + k] * z;
                        k += 1;
                    }
                } else {
                    // Differential Evolution: the proposal increment is the
                    // scaled difference of two other chains' batch-start
                    // states, read from the frozen `init` snapshot, plus a
                    // small per-dimension jitter. The pair is redrawn each
                    // iteration; with probability 0.1 the scale collapses to
                    // 1.0 for an occasional mode-crossing jump (ter Braak 2006).
                    ctr += 1u32;
                    let mut a = usize::cast_from(
                        rand_uniform(seedmix, ctr) * f32::cast_from(n_chains));
                    if a >= n_chains {
                        a = n_chains - 1usize;
                    }
                    ctr += 1u32;
                    let mut b = usize::cast_from(
                        rand_uniform(seedmix, ctr) * f32::cast_from(n_chains));
                    if b >= n_chains {
                        b = n_chains - 1usize;
                    }
                    if b == a {
                        b += 1usize;
                        if b >= n_chains {
                            b = 0usize;
                        }
                    }
                    ctr += 1u32;
                    let mut g = de_params[0];
                    if rand_uniform(seedmix, ctr) < 0.1 {
                        g = 1.0f32;
                    }
                    let noise = de_params[1];
                    let abase = a * n_params;
                    let bbase = b * n_params;
                    let mut k = 0usize;
                    while k < n_params {
                        ctr += 1u32;
                        let u1 = rand_uniform(seedmix, ctr);
                        ctr += 1u32;
                        let u2 = rand_uniform(seedmix, ctr);
                        let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                        let diff = init[abase + k] - init[bbase + k];
                        prop[k] =
                            state[k] + g * diff + noise * proposal_sd[pbase + k] * z;
                        k += 1;
                    }
                }
            }
            sync_cube();

            // Block-parallel log-likelihood of the proposal.
            let mut pk = 0usize;
            while pk < n_params {
                local[pk] = prop[pk];
                pk += 1;
            }
            let mut ppart = 0.0f32;
            let mut pobs = tid;
            while pobs < n_obs {
                ppart += vm_eval(code, consts, n_instr, &local, data,
                                 pobs * n_cols);
                pobs += block;
            }
            partials[tid] = ppart;
            sync_cube();
            let mut s = block / 2usize;
            while s >= 1usize {
                if tid < s {
                    partials[tid] = partials[tid] + partials[tid + s];
                }
                sync_cube();
                s /= 2usize;
            }

            // Thread 0 accepts or rejects and records the draw. The
            // acceptance ratio is divided by the chain temperature; for
            // T = 1 this recovers the textbook Metropolis ratio, and for
            // T > 1 the chain accepts more aggressively, the cornerstone
            // of parallel tempering's hot chains.
            if tid == 0 {
                let plp = partials[0] + vm_eval(prior_code, prior_consts,
                                                prior_n, &local, data, 0);
                ctr += 1u32;
                let u = rand_uniform(seedmix, ctr);
                let temperature = temperatures[chain];
                if u.ln() < (plp - cur) / temperature {
                    let mut m = 0usize;
                    while m < n_params {
                        state[m] = prop[m];
                        m += 1;
                    }
                    cur = plp;
                    accepts += 1u32;
                }
                let mut m = 0usize;
                while m < n_params {
                    out_draws[t + n_iter * (chain + n_chains * m)] = state[m];
                    m += 1;
                }
            }
            sync_cube();
            t += 1;
        }
        if tid == 0 {
            out_accept[chain] = f32::cast_from(accepts) / f32::cast_from(n_iter);
            out_logpost[chain] = cur;
        }
    }
}

/// Result of a whole-chain Metropolis run.
pub struct ChainResult {
    pub n_iter: usize,
    pub n_chains: usize,
    pub n_params: usize,
    pub draws: Vec<f32>,
    pub accept_rate: Vec<f32>,
    /// Raw log-posterior at the final state of each chain, needed by the
    /// parallel-tempering swap step which compares densities across chains.
    pub last_log_post: Vec<f64>,
}

#[allow(clippy::too_many_arguments, dead_code)]
fn run_block<R: Runtime>(
    code: &[u32],
    consts: &[f32],
    prior_code: &[u32],
    prior_consts: &[f32],
    n_params: usize,
    data: &[f32],
    n_cols: usize,
    n_obs: usize,
    init: &[f32],
    proposal_sd: &[f32],
    temperatures: &[f32],
    n_iter: usize,
    seed: u32,
    proposal_mode: u32,
    de_params: &[f32],
) -> ChainResult {
    let n_chains = init.len() / n_params;
    let client = R::client(&Default::default());

    let code_h = client.create_from_slice(u32::as_bytes(code));
    let consts_h = client.create_from_slice(f32::as_bytes(consts));
    let prior_code_h = client.create_from_slice(u32::as_bytes(prior_code));
    let prior_consts_h = client.create_from_slice(f32::as_bytes(prior_consts));
    let data_h = client.create_from_slice(f32::as_bytes(data));
    let init_h = client.create_from_slice(f32::as_bytes(init));
    let psd_h = client.create_from_slice(f32::as_bytes(proposal_sd));
    let temp_h = client.create_from_slice(f32::as_bytes(temperatures));
    let de_h = client.create_from_slice(f32::as_bytes(de_params));
    let meta: [u32; 8] = [
        (code.len() / 2) as u32,
        n_params as u32,
        n_cols as u32,
        n_obs as u32,
        n_iter as u32,
        (prior_code.len() / 2) as u32,
        seed,
        proposal_mode,
    ];
    let meta_h = client.create_from_slice(u32::as_bytes(&meta));

    let f32_size = core::mem::size_of::<f32>();
    let draws_h = client.empty(n_iter * n_chains * n_params * f32_size);
    let accept_h = client.empty(n_chains * f32_size);
    let logpost_h = client.empty(n_chains * f32_size);

    // One block per chain, 256 threads per block.
    let count = CubeCount::Static(n_chains as u32, 1, 1);
    let dim = CubeDim { x: 256, y: 1, z: 1 };

    unsafe {
        metropolis_block_kernel::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(code_h, code.len()),
            ArrayArg::from_raw_parts(consts_h, consts.len().max(1)),
            ArrayArg::from_raw_parts(prior_code_h, prior_code.len().max(1)),
            ArrayArg::from_raw_parts(prior_consts_h, prior_consts.len().max(1)),
            ArrayArg::from_raw_parts(data_h, data.len().max(1)),
            ArrayArg::from_raw_parts(init_h, init.len()),
            ArrayArg::from_raw_parts(psd_h, proposal_sd.len()),
            ArrayArg::from_raw_parts(temp_h, temperatures.len()),
            ArrayArg::from_raw_parts(de_h, de_params.len().max(1)),
            ArrayArg::from_raw_parts(meta_h, meta.len()),
            ArrayArg::from_raw_parts(draws_h.clone(), n_iter * n_chains * n_params),
            ArrayArg::from_raw_parts(accept_h.clone(), n_chains),
            ArrayArg::from_raw_parts(logpost_h.clone(), n_chains),
        );
    }

    let draws = f32::from_bytes(&client.read_one_unchecked(draws_h)).to_vec();
    let accept_rate =
        f32::from_bytes(&client.read_one_unchecked(accept_h)).to_vec();
    let last_log_post: Vec<f64> =
        f32::from_bytes(&client.read_one_unchecked(logpost_h))
            .iter()
            .map(|&v| v as f64)
            .collect();

    ChainResult {
        n_iter,
        n_chains,
        n_params,
        draws,
        accept_rate,
        last_log_post,
    }
}

/// Run the whole-chain Metropolis sampler on the chosen backend. The float
/// inputs are f64; the CPU backend keeps f64 throughout, the GPU backends cast
/// to f32 for the device kernel.
#[allow(clippy::too_many_arguments)]
pub fn gpu_metropolis_chain(
    code: &[u32],
    consts: &[f64],
    prior_code: &[u32],
    prior_consts: &[f64],
    n_params: usize,
    data: &[f64],
    n_cols: usize,
    n_obs: usize,
    init: &[f64],
    proposal_sd: &[f64],
    temperatures: &[f64],
    n_iter: usize,
    seed: u32,
    backend: crate::gpu::Backend,
    proposal_mode: u32,
    gamma: f64,
    de_noise: f64,
) -> ChainResult {
    // The CPU backend uses a native Rust implementation in f64; the CubeCL CPU
    // runtime is slow for this interpreter-style kernel.
    if backend == crate::gpu::Backend::Cpu {
        return crate::cpu_native::run(
            code, consts, prior_code, prior_consts, n_params, data, n_cols,
            n_obs, init, proposal_sd, temperatures, n_iter, seed,
            proposal_mode, gamma, de_noise,
        );
    }
    // GPU backends are compiled in only when the matching Cargo feature is on.
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    {
        let f32v = |s: &[f64]| -> Vec<f32> {
            s.iter().map(|&v| v as f32).collect()
        };
        let (c, pc, d, i, p, tmp) = (
            f32v(consts), f32v(prior_consts), f32v(data), f32v(init),
            f32v(proposal_sd), f32v(temperatures),
        );
        let de_params: [f32; 2] = [gamma as f32, de_noise as f32];
        match backend {
            #[cfg(feature = "cuda")]
            crate::gpu::Backend::Cuda => {
                return run_block::<cubecl::cuda::CudaRuntime>(
                    code, &c, prior_code, &pc, n_params, &d, n_cols, n_obs,
                    &i, &p, &tmp, n_iter, seed, proposal_mode, &de_params,
                );
            }
            #[cfg(feature = "vulkan")]
            crate::gpu::Backend::Vulkan => {
                return run_block::<cubecl::wgpu::WgpuRuntime>(
                    code, &c, prior_code, &pc, n_params, &d, n_cols, n_obs,
                    &i, &p, &tmp, n_iter, seed, proposal_mode, &de_params,
                );
            }
            _ => {}
        }
    }
    panic!(
        "GPU backend not available in this build; rebuild gpumetropolis with \
         the 'cuda' or 'vulkan' Cargo feature"
    )
}
