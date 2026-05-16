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
    meta: &Array<u32>,        // [n_instr, n_params, n_cols, n_obs, n_iter, prior_n, seed]
    out_draws: &mut Array<f32>,
    out_accept: &mut Array<f32>,
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
        let pbase = chain * n_params;
        let seedmix = seed + u32::cast_from(chain) * 2654435761u32;

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
                let mut k = 0usize;
                while k < n_params {
                    ctr += 1u32;
                    let u1 = rand_uniform(seedmix, ctr);
                    ctr += 1u32;
                    let u2 = rand_uniform(seedmix, ctr);
                    let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                    prop[k] = state[k] + proposal_sd[k] * z;
                    k += 1;
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

            // Thread 0 accepts or rejects and records the draw.
            if tid == 0 {
                let plp = partials[0] + vm_eval(prior_code, prior_consts,
                                                prior_n, &local, data, 0);
                ctr += 1u32;
                let u = rand_uniform(seedmix, ctr);
                if u.ln() < plp - cur {
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
}

#[allow(clippy::too_many_arguments)]
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
    n_iter: usize,
    seed: u32,
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
    let meta: [u32; 7] = [
        (code.len() / 2) as u32,
        n_params as u32,
        n_cols as u32,
        n_obs as u32,
        n_iter as u32,
        (prior_code.len() / 2) as u32,
        seed,
    ];
    let meta_h = client.create_from_slice(u32::as_bytes(&meta));

    let f32_size = core::mem::size_of::<f32>();
    let draws_h = client.empty(n_iter * n_chains * n_params * f32_size);
    let accept_h = client.empty(n_chains * f32_size);

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
            ArrayArg::from_raw_parts(meta_h, meta.len()),
            ArrayArg::from_raw_parts(draws_h.clone(), n_iter * n_chains * n_params),
            ArrayArg::from_raw_parts(accept_h.clone(), n_chains),
        );
    }

    let draws = f32::from_bytes(&client.read_one_unchecked(draws_h)).to_vec();
    let accept_rate =
        f32::from_bytes(&client.read_one_unchecked(accept_h)).to_vec();

    ChainResult {
        n_iter,
        n_chains,
        n_params,
        draws,
        accept_rate,
    }
}

/// Run the whole-chain Metropolis sampler on the chosen backend. The CPU
/// backend uses a native Rust implementation; the GPU backends use the
/// block-per-chain kernel.
#[allow(clippy::too_many_arguments)]
pub fn gpu_metropolis_chain(
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
    n_iter: usize,
    seed: u32,
    backend: crate::gpu::Backend,
) -> ChainResult {
    match backend {
        crate::gpu::Backend::Cpu => crate::cpu_native::run(
            code, consts, prior_code, prior_consts, n_params, data, n_cols,
            n_obs, init, proposal_sd, n_iter, seed,
        ),
        crate::gpu::Backend::Cuda => run_block::<cubecl::cuda::CudaRuntime>(
            code, consts, prior_code, prior_consts, n_params, data, n_cols,
            n_obs, init, proposal_sd, n_iter, seed,
        ),
        crate::gpu::Backend::Vulkan => run_block::<cubecl::wgpu::WgpuRuntime>(
            code, consts, prior_code, prior_consts, n_params, data, n_cols,
            n_obs, init, proposal_sd, n_iter, seed,
        ),
    }
}
