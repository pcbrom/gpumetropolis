//! Native CPU backend.
//!
//! The whole-chain Metropolis sampler in plain Rust, parallel over chains with
//! rayon. The CubeCL CPU runtime is slow for this interpreter-style kernel; a
//! native compiled loop with one thread per chain group is far faster. The
//! RNG, the bytecode VM and the arithmetic mirror `chain_kernel.rs` exactly, so
//! the CPU backend and the GPU backends sample the same way.

use rayon::prelude::*;

use crate::chain_kernel::ChainResult;

/// triple32 integer hash (Wellons).
fn triple32(mut h: u32) -> u32 {
    h ^= h >> 17;
    h = h.wrapping_mul(2129725213);
    h ^= h >> 11;
    h = h.wrapping_mul(2890025505);
    h ^= h >> 15;
    h = h.wrapping_mul(831628273);
    h ^= h >> 14;
    h
}

#[inline]
fn rand_uniform(seedmix: u32, ctr: u32) -> f32 {
    (triple32(seedmix.wrapping_add(ctr)) as f32 + 0.5) / 4294967296.0
}

/// Run one bytecode program once; an empty program returns zero.
fn vm_eval(code: &[u32], consts: &[f32], params: &[f32], data: &[f32],
           data_base: usize) -> f32 {
    if code.is_empty() {
        return 0.0;
    }
    let mut stack = [0.0f32; 32];
    let mut sp = 0usize;
    for pc in 0..(code.len() / 2) {
        let op = code[2 * pc];
        let arg = code[2 * pc + 1] as usize;
        match op {
            0 => { stack[sp] = consts[arg]; sp += 1; }
            1 => { stack[sp] = params[arg]; sp += 1; }
            2 => { stack[sp] = data[data_base + arg]; sp += 1; }
            3 => { stack[sp - 2] += stack[sp - 1]; sp -= 1; }
            4 => { stack[sp - 2] -= stack[sp - 1]; sp -= 1; }
            5 => { stack[sp - 2] *= stack[sp - 1]; sp -= 1; }
            6 => { stack[sp - 2] /= stack[sp - 1]; sp -= 1; }
            7 => { stack[sp - 1] = -stack[sp - 1]; }
            8 => { stack[sp - 1] = stack[sp - 1].exp(); }
            9 => { stack[sp - 1] = stack[sp - 1].ln(); }
            10 => { stack[sp - 1] = stack[sp - 1].sqrt(); }
            _ => { stack[sp - 2] = stack[sp - 2].powf(stack[sp - 1]); sp -= 1; }
        }
    }
    stack[0]
}

#[allow(clippy::too_many_arguments)]
fn log_post(code: &[u32], consts: &[f32], prior_code: &[u32],
            prior_consts: &[f32], params: &[f32], data: &[f32],
            n_cols: usize, n_obs: usize) -> f32 {
    let mut acc = 0.0f32;
    for obs in 0..n_obs {
        acc += vm_eval(code, consts, params, data, obs * n_cols);
    }
    acc + vm_eval(prior_code, prior_consts, params, &[], 0)
}

/// Run the whole-chain Metropolis sampler natively, parallel over chains.
#[allow(clippy::too_many_arguments)]
pub fn run(
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
    let two_pi = 6.2831853f32;

    // Each chain produces its own draws and acceptance rate; assembled after.
    let per_chain: Vec<(Vec<f32>, f32)> = (0..n_chains)
        .into_par_iter()
        .map(|c| {
            let pbase = c * n_params;
            let seedmix = seed.wrapping_add((c as u32).wrapping_mul(2654435761));
            let mut state: Vec<f32> = init[pbase..pbase + n_params].to_vec();
            let mut prop = state.clone();
            let mut cur = log_post(code, consts, prior_code, prior_consts,
                                   &state, data, n_cols, n_obs);
            let mut ctr = 0u32;
            let mut accepts = 0u32;
            let mut cdraws = vec![0.0f32; n_iter * n_params];

            for t in 0..n_iter {
                for j in 0..n_params {
                    ctr += 1;
                    let u1 = rand_uniform(seedmix, ctr);
                    ctr += 1;
                    let u2 = rand_uniform(seedmix, ctr);
                    let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                    prop[j] = state[j] + proposal_sd[j] * z;
                }
                let plp = log_post(code, consts, prior_code, prior_consts,
                                   &prop, data, n_cols, n_obs);
                ctr += 1;
                let u = rand_uniform(seedmix, ctr);
                if u.ln() < plp - cur {
                    state.copy_from_slice(&prop);
                    cur = plp;
                    accepts += 1;
                }
                for j in 0..n_params {
                    cdraws[t * n_params + j] = state[j];
                }
            }
            let rate = if n_iter == 0 {
                0.0
            } else {
                accepts as f32 / n_iter as f32
            };
            (cdraws, rate)
        })
        .collect();

    // Assemble into the (n_iter, n_chains, n_params) column-major layout.
    let mut draws = vec![0.0f32; n_iter * n_chains * n_params];
    let mut accept_rate = vec![0.0f32; n_chains];
    for (c, (cdraws, rate)) in per_chain.iter().enumerate() {
        accept_rate[c] = *rate;
        for t in 0..n_iter {
            for j in 0..n_params {
                draws[t + n_iter * (c + n_chains * j)] = cdraws[t * n_params + j];
            }
        }
    }

    ChainResult { n_iter, n_chains, n_params, draws, accept_rate }
}
