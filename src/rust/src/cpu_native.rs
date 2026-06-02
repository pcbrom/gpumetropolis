//! Native CPU backend.
//!
//! The whole-chain Metropolis sampler in plain Rust, parallel over chains with
//! rayon. The log-likelihood is JIT-compiled to native code (see `jit.rs`),
//! with an interpreter fallback. The CPU path works in f64: the log-density
//! sums many terms, and f32 loses the small difference that drives the
//! Metropolis acceptance once the data set is large.

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
fn rand_uniform(seedmix: u32, ctr: u32) -> f64 {
    (triple32(seedmix.wrapping_add(ctr)) as f64 + 0.5) / 4294967296.0
}

/// Run one bytecode program once; an empty program returns zero.
fn vm_eval(code: &[u32], consts: &[f64], params: &[f64], data: &[f64],
           data_base: usize) -> f64 {
    if code.is_empty() {
        return 0.0;
    }
    let mut stack = [0.0f64; 32];
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

/// Interpreted log-likelihood sum over the observations, the fallback path.
fn interp_loglik(code: &[u32], consts: &[f64], params: &[f64], data: &[f64],
                 n_cols: usize, n_obs: usize) -> f64 {
    let mut acc = 0.0f64;
    for obs in 0..n_obs {
        acc += vm_eval(code, consts, params, data, obs * n_cols);
    }
    acc
}

/// Run the whole-chain Metropolis sampler natively, parallel over chains.
#[allow(clippy::too_many_arguments)]
pub fn run(
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
    n_iter: usize,
    seed: u32,
) -> ChainResult {
    let n_chains = init.len() / n_params;
    let two_pi = 6.283185307179586f64;

    // Compile the log-likelihood to native code once; on failure use the
    // interpreter.
    let jit = crate::jit::compile_loglik(code, consts).ok();
    let log_post = |params: &[f64]| -> f64 {
        let data_ll = match &jit {
            Some(j) => j.eval(params, data, n_obs, n_cols),
            None => interp_loglik(code, consts, params, data, n_cols, n_obs),
        };
        data_ll + vm_eval(prior_code, prior_consts, params, &[], 0)
    };

    // Each chain produces its own draws (f32) and acceptance rate.
    let per_chain: Vec<(Vec<f32>, f32)> = (0..n_chains)
        .into_par_iter()
        .map(|c| {
            let pbase = c * n_params;
            // The base seed is hashed before the chain offset is added, so two
            // runs with consecutive integer seeds get well-separated counter
            // streams rather than streams that overlap by a one-counter shift.
            let seedmix =
                triple32(seed).wrapping_add((c as u32).wrapping_mul(2654435761));
            let mut state: Vec<f64> = init[pbase..pbase + n_params].to_vec();
            let mut prop = state.clone();
            let mut cur = log_post(&state);
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
                    prop[j] = state[j] + proposal_sd[pbase + j] * z;
                }
                let plp = log_post(&prop);
                ctr += 1;
                let u = rand_uniform(seedmix, ctr);
                if u.ln() < plp - cur {
                    state.copy_from_slice(&prop);
                    cur = plp;
                    accepts += 1;
                }
                for j in 0..n_params {
                    cdraws[t * n_params + j] = state[j] as f32;
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
