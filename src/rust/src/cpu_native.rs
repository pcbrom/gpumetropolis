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

/// Scratch buffers for the reverse-mode tape, sized to the program length
/// and reused across observations and iterations so the gradient path does
/// not allocate in the hot loop.
struct GradWork {
    vals: Vec<f64>,
    adj: Vec<f64>,
    lhs: Vec<u32>,
    rhs: Vec<u32>,
}

impl GradWork {
    fn new(max_ops: usize) -> Self {
        GradWork {
            vals: vec![0.0; max_ops],
            adj: vec![0.0; max_ops],
            lhs: vec![u32::MAX; max_ops],
            rhs: vec![u32::MAX; max_ops],
        }
    }
}

/// Reverse-mode automatic differentiation of one bytecode program at one
/// observation. The forward pass executes the stack machine while recording
/// a tape: the value each instruction produced and the tape indices of its
/// operands (the instruction index doubles as the tape index, since every
/// instruction produces exactly one value). The backward pass walks the tape
/// in reverse propagating adjoints through the textbook local derivatives,
/// and deposits d(value)/d(params[k]) into `grad[k]`. Returns the value, so
/// one pass yields both the density and its gradient.
fn vm_eval_grad(code: &[u32], consts: &[f64], params: &[f64], data: &[f64],
                data_base: usize, work: &mut GradWork, grad: &mut [f64])
                -> f64 {
    if code.is_empty() {
        return 0.0;
    }
    let n_ops = code.len() / 2;
    let mut istack = [0u32; 32];
    let mut sp = 0usize;
    for pc in 0..n_ops {
        let op = code[2 * pc];
        let arg = code[2 * pc + 1] as usize;
        match op {
            0 => {
                work.vals[pc] = consts[arg];
                work.lhs[pc] = u32::MAX;
                istack[sp] = pc as u32;
                sp += 1;
            }
            1 => {
                work.vals[pc] = params[arg];
                work.lhs[pc] = u32::MAX;
                istack[sp] = pc as u32;
                sp += 1;
            }
            2 => {
                work.vals[pc] = data[data_base + arg];
                work.lhs[pc] = u32::MAX;
                istack[sp] = pc as u32;
                sp += 1;
            }
            7..=10 => {
                let l = istack[sp - 1] as usize;
                let x = work.vals[l];
                work.vals[pc] = match op {
                    7 => -x,
                    8 => x.exp(),
                    9 => x.ln(),
                    _ => x.sqrt(),
                };
                work.lhs[pc] = l as u32;
                work.rhs[pc] = u32::MAX;
                istack[sp - 1] = pc as u32;
            }
            _ => {
                let r = istack[sp - 1] as usize;
                let l = istack[sp - 2] as usize;
                let a = work.vals[l];
                let b = work.vals[r];
                work.vals[pc] = match op {
                    3 => a + b,
                    4 => a - b,
                    5 => a * b,
                    6 => a / b,
                    _ => a.powf(b),
                };
                work.lhs[pc] = l as u32;
                work.rhs[pc] = r as u32;
                istack[sp - 2] = pc as u32;
                sp -= 1;
            }
        }
    }
    let root = n_ops - 1;
    for slot in work.adj[..n_ops].iter_mut() {
        *slot = 0.0;
    }
    work.adj[root] = 1.0;
    for pc in (0..n_ops).rev() {
        let g = work.adj[pc];
        if g == 0.0 {
            continue;
        }
        let op = code[2 * pc];
        let arg = code[2 * pc + 1] as usize;
        match op {
            0 | 2 => {}
            1 => {
                grad[arg] += g;
            }
            3 => {
                work.adj[work.lhs[pc] as usize] += g;
                work.adj[work.rhs[pc] as usize] += g;
            }
            4 => {
                work.adj[work.lhs[pc] as usize] += g;
                work.adj[work.rhs[pc] as usize] -= g;
            }
            5 => {
                let l = work.lhs[pc] as usize;
                let r = work.rhs[pc] as usize;
                work.adj[l] += g * work.vals[r];
                work.adj[r] += g * work.vals[l];
            }
            6 => {
                let l = work.lhs[pc] as usize;
                let r = work.rhs[pc] as usize;
                work.adj[l] += g / work.vals[r];
                work.adj[r] -= g * work.vals[pc] / work.vals[r];
            }
            7 => {
                work.adj[work.lhs[pc] as usize] -= g;
            }
            8 => {
                work.adj[work.lhs[pc] as usize] += g * work.vals[pc];
            }
            9 => {
                let l = work.lhs[pc] as usize;
                work.adj[l] += g / work.vals[l];
            }
            10 => {
                work.adj[work.lhs[pc] as usize] += g * 0.5 / work.vals[pc];
            }
            _ => {
                let l = work.lhs[pc] as usize;
                let r = work.rhs[pc] as usize;
                let a = work.vals[l];
                let b = work.vals[r];
                work.adj[l] += g * b * a.powf(b - 1.0);
                if a > 0.0 {
                    work.adj[r] += g * work.vals[pc] * a.ln();
                }
            }
        }
    }
    work.vals[root]
}

/// Log-posterior and its gradient in one pass: the data term summed over the
/// observations plus the prior term, each through the reverse-mode tape.
/// `grad` is overwritten. Returns the log-posterior value.
#[allow(clippy::too_many_arguments)]
fn logpost_grad(code: &[u32], consts: &[f64], prior_code: &[u32],
                prior_consts: &[f64], params: &[f64], data: &[f64],
                n_cols: usize, n_obs: usize, work: &mut GradWork,
                grad: &mut [f64]) -> f64 {
    for slot in grad.iter_mut() {
        *slot = 0.0;
    }
    let mut acc = 0.0f64;
    for obs in 0..n_obs {
        acc += vm_eval_grad(code, consts, params, data, obs * n_cols, work,
                            grad);
    }
    acc + vm_eval_grad(prior_code, prior_consts, params, &[], 0, work, grad)
}

/// Gradient of the log-likelihood (data term only), summed over the
/// observations, at each of the `points.len() / n_params` parameter vectors.
/// Point-major flat output. Backs host-side gradient checks and the exact
/// observed-information path of the Cramer-Rao diagnostic (central
/// differences of the gradient rather than second differences of the value).
#[allow(clippy::too_many_arguments)]
pub fn grad_batch(
    code: &[u32],
    consts: &[f64],
    n_params: usize,
    data: &[f64],
    n_cols: usize,
    n_obs: usize,
    points: &[f64],
) -> Vec<f64> {
    let n_points = if n_params == 0 { 0 } else { points.len() / n_params };
    let n_ops = (code.len() / 2).max(1);
    let mut out = vec![0.0f64; n_points * n_params];
    out.par_chunks_mut(n_params.max(1))
        .enumerate()
        .for_each(|(p, gslot)| {
            let params = &points[p * n_params..(p + 1) * n_params];
            let mut work = GradWork::new(n_ops);
            for obs in 0..n_obs {
                vm_eval_grad(code, consts, params, data, obs * n_cols,
                             &mut work, gslot);
            }
        });
    out
}

/// Evaluate the compiled log-likelihood, summed over observations, at each of
/// the `points.len() / n_params` parameter vectors, returned in order. Reuses
/// the JIT path with the interpreter fallback. This backs the host-side
/// observed Fisher information / Cramer-Rao diagnostic, which needs the
/// log-likelihood at a small stencil of points around the posterior mean; it
/// is the same compiled log-likelihood the sampler evaluates, so the curvature
/// it returns is consistent with the draws.
#[allow(clippy::too_many_arguments)]
pub fn loglik_batch(
    code: &[u32],
    consts: &[f64],
    n_params: usize,
    data: &[f64],
    n_cols: usize,
    n_obs: usize,
    points: &[f64],
) -> Vec<f64> {
    let n_points = if n_params == 0 { 0 } else { points.len() / n_params };
    let jit = crate::jit::compile_loglik(code, consts).ok();
    (0..n_points)
        .map(|p| {
            let pb = p * n_params;
            let params = &points[pb..pb + n_params];
            match &jit {
                Some(j) => j.eval(params, data, n_obs, n_cols),
                None => interp_loglik(code, consts, params, data, n_cols, n_obs),
            }
        })
        .collect()
}

/// Evaluate the compiled log-likelihood per observation at each of the
/// `points.len() / n_params` parameter vectors. Returns a flat point-major
/// then observation-major buffer, `out[p * n_obs + i]` the log-likelihood of
/// observation `i` at point `p`. This backs the WAIC and PSIS-LOO model
/// comparison, which need the pointwise log-likelihood matrix rather than the
/// summed value. The interpreter is used per observation because the JIT path
/// returns the sum over observations, not the per-observation terms; the work
/// is parallel over points.
#[allow(clippy::too_many_arguments)]
pub fn loglik_pointwise(
    code: &[u32],
    consts: &[f64],
    n_params: usize,
    data: &[f64],
    n_cols: usize,
    n_obs: usize,
    points: &[f64],
) -> Vec<f64> {
    let n_points = if n_params == 0 { 0 } else { points.len() / n_params };
    let mut out = vec![0.0f64; n_points * n_obs];
    out.par_chunks_mut(n_obs.max(1))
        .enumerate()
        .for_each(|(p, row)| {
            let params = &points[p * n_params..(p + 1) * n_params];
            for (i, slot) in row.iter_mut().enumerate().take(n_obs) {
                *slot = vm_eval(code, consts, params, data, i * n_cols);
            }
        });
    out
}

/// Synchronous Differential Evolution MCMC (path B): the population advances
/// one generation at a time behind a barrier, with a double buffer so every
/// chain's proposal in generation `t + 1` reads the whole population of
/// generation `t`. This is the canonical per-generation mixing of ter Braak
/// (2006); the batched path A instead freezes the pool for a whole batch.
/// The difference pair excludes the proposing chain, per the standard scheme.
#[allow(clippy::too_many_arguments)]
pub fn run_de_sync(
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
    gamma: f64,
    de_noise: f64,
) -> ChainResult {
    let n_chains = init.len() / n_params;
    let two_pi = 6.283185307179586f64;

    let jit = crate::jit::compile_loglik(code, consts).ok();
    let log_post = |params: &[f64]| -> f64 {
        let data_ll = match &jit {
            Some(j) => j.eval(params, data, n_obs, n_cols),
            None => interp_loglik(code, consts, params, data, n_cols, n_obs),
        };
        data_ll + vm_eval(prior_code, prior_consts, params, &[], 0)
    };

    let mut pop_cur: Vec<f64> = init.to_vec();
    let mut pop_next: Vec<f64> = vec![0.0; pop_cur.len()];
    let mut cur_lp: Vec<f64> = (0..n_chains)
        .into_par_iter()
        .map(|c| log_post(&pop_cur[c * n_params..(c + 1) * n_params]))
        .collect();
    let mut ctrs: Vec<u32> = vec![0; n_chains];
    let mut accepts: Vec<u32> = vec![0; n_chains];
    let seedmixes: Vec<u32> = (0..n_chains)
        .map(|c| triple32(seed).wrapping_add((c as u32).wrapping_mul(2654435761)))
        .collect();
    let mut draws = vec![0.0f32; n_iter * n_chains * n_params];

    for t in 0..n_iter {
        // One generation: every chain proposes against the generation-t pool
        // and writes its new state into the generation-(t+1) buffer. The
        // rayon pass is the barrier.
        let gen: Vec<(Vec<f64>, f64, u32, bool)> = (0..n_chains)
            .into_par_iter()
            .map(|c| {
                let pbase = c * n_params;
                let seedmix = seedmixes[c];
                let mut ctr = ctrs[c];
                let state = &pop_cur[pbase..pbase + n_params];
                // Pair (a, b) from the current generation, excluding self.
                ctr += 1;
                let mut a = (rand_uniform(seedmix, ctr) * n_chains as f64) as usize;
                if a >= n_chains { a = n_chains - 1; }
                if a == c { a = (a + 1) % n_chains; }
                ctr += 1;
                let mut b = (rand_uniform(seedmix, ctr) * n_chains as f64) as usize;
                if b >= n_chains { b = n_chains - 1; }
                while b == c || b == a { b = (b + 1) % n_chains; }
                ctr += 1;
                let g = if rand_uniform(seedmix, ctr) < 0.1 { 1.0 } else { gamma };
                let abase = a * n_params;
                let bbase = b * n_params;
                let mut prop = vec![0.0f64; n_params];
                for j in 0..n_params {
                    ctr += 1;
                    let u1 = rand_uniform(seedmix, ctr);
                    ctr += 1;
                    let u2 = rand_uniform(seedmix, ctr);
                    let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                    let diff = pop_cur[abase + j] - pop_cur[bbase + j];
                    prop[j] = state[j] + g * diff
                        + de_noise * proposal_sd[pbase + j] * z;
                }
                let plp = log_post(&prop);
                ctr += 1;
                let u = rand_uniform(seedmix, ctr);
                if u.ln() < plp - cur_lp[c] {
                    (prop, plp, ctr, true)
                } else {
                    (state.to_vec(), cur_lp[c], ctr, false)
                }
            })
            .collect();

        for (c, (state, lp, ctr, accepted)) in gen.into_iter().enumerate() {
            let pbase = c * n_params;
            pop_next[pbase..pbase + n_params].copy_from_slice(&state);
            cur_lp[c] = lp;
            ctrs[c] = ctr;
            if accepted { accepts[c] += 1; }
            for j in 0..n_params {
                draws[t + n_iter * (c + n_chains * j)] = state[j] as f32;
            }
        }
        std::mem::swap(&mut pop_cur, &mut pop_next);
    }

    let accept_rate: Vec<f32> = accepts
        .iter()
        .map(|&a| if n_iter == 0 { 0.0 } else { a as f32 / n_iter as f32 })
        .collect();
    let last_log_post = cur_lp;
    ChainResult { n_iter, n_chains, n_params, draws, accept_rate, last_log_post }
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
    temperatures: &[f64],
    n_iter: usize,
    seed: u32,
    proposal_mode: u32,
    gamma: f64,
    de_noise: f64,
    proposal_l: &[f64],
) -> ChainResult {
    let n_chains = init.len() / n_params;
    let two_pi = 6.283185307179586f64;

    // Compile the log-likelihood to native code once; on failure use the
    // interpreter. Under MALA the gradient is also compiled: the reverse-mode
    // tape unrolls to straight-line native code, so a Langevin step costs a
    // small multiple of a plain evaluation instead of an interpreter pass.
    let jit = crate::jit::compile_loglik(code, consts).ok();
    let jit_grad = if proposal_mode == 3 {
        crate::jit::compile_grad(code, consts, n_params).ok()
    } else {
        None
    };
    let log_post = |params: &[f64]| -> f64 {
        let data_ll = match &jit {
            Some(j) => j.eval(params, data, n_obs, n_cols),
            None => interp_loglik(code, consts, params, data, n_cols, n_obs),
        };
        data_ll + vm_eval(prior_code, prior_consts, params, &[], 0)
    };

    // Each chain produces its own draws (f32), an acceptance rate, and the
    // raw log-posterior at its final state. The latter is needed by the
    // parallel tempering swap step, which compares densities across chains.
    let per_chain: Vec<(Vec<f32>, f32, f64)> = (0..n_chains)
        .into_par_iter()
        .map(|c| {
            let pbase = c * n_params;
            let temperature = temperatures[c];
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

            let lbase = c * n_params * n_params;
            let mut zbuf = vec![0.0f64; n_params];

            // MALA (proposal_mode 3) state: the gradient at the current
            // state is cached across iterations (it only changes on accept),
            // and the tape buffers are reused. The preconditioner is
            // M = Ls Ls' with Ls the per-chain scaled Cholesky factor in
            // `proposal_l`; with an empty `proposal_l` the factor is the
            // diagonal `proposal_sd`.
            let mala = proposal_mode == 3;
            let n_ops = (code.len() / 2).max(prior_code.len() / 2).max(1);
            let mut work = GradWork::new(n_ops);
            let mut gx = vec![0.0f64; n_params];
            let mut gy = vec![0.0f64; n_params];
            let mut mean_x = vec![0.0f64; n_params];
            let mut mean_y = vec![0.0f64; n_params];
            let mut vbuf = vec![0.0f64; n_params];
            // Value and gradient in one pass: the compiled data term plus
            // the interpreted prior term (a handful of ops, no data loop).
            let mut lp_grad = |params: &[f64], grad: &mut [f64],
                               work: &mut GradWork| -> f64 {
                match &jit_grad {
                    Some(jg) => {
                        let v = jg.eval_grad(params, data, n_obs, n_cols,
                                             grad);
                        v + vm_eval_grad(prior_code, prior_consts, params,
                                         &[], 0, work, grad)
                    }
                    None => logpost_grad(code, consts, prior_code,
                                         prior_consts, params, data, n_cols,
                                         n_obs, work, grad),
                }
            };
            if mala {
                cur = lp_grad(&state, &mut gx, &mut work);
            }
            // Preconditioned, truncated Langevin drift:
            // out = f * 0.5 * (Ls Ls') g, computed as Ls (0.5 Ls' g) with
            // the lower-triangular factor, or through the diagonal scale
            // when no factor is supplied. The truncation caps the whitened
            // drift norm at `2 sqrt(d)` (MALTA, Roberts and Tweedie 1996):
            // far from the mode the raw gradient explodes, the untruncated
            // proposal overshoots, every step is rejected and the chain
            // freezes; the cap bounds the step while leaving the
            // equilibrium drift, whose whitened norm is O(sqrt(d)),
            // untouched. The Metropolis correction stays exact because the
            // proposal density uses the same truncated mean.
            let drift_max = 2.0 * (n_params as f64).sqrt();
            let half_m_times = |g: &[f64], out: &mut [f64], tmp: &mut [f64]| {
                if proposal_l.is_empty() {
                    for j in 0..n_params {
                        tmp[j] = 0.5 * proposal_sd[pbase + j] * g[j];
                    }
                } else {
                    for j in 0..n_params {
                        let mut acc = 0.0;
                        for k in j..n_params {
                            acc += proposal_l[lbase + k * n_params + j] * g[k];
                        }
                        tmp[j] = 0.5 * acc;
                    }
                }
                let norm = tmp[..n_params]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                let f = if norm > drift_max { drift_max / norm } else { 1.0 };
                if proposal_l.is_empty() {
                    for j in 0..n_params {
                        out[j] = f * proposal_sd[pbase + j] * tmp[j];
                    }
                } else {
                    for k in 0..n_params {
                        let mut acc = 0.0;
                        for j in 0..=k {
                            acc += proposal_l[lbase + k * n_params + j] * tmp[j];
                        }
                        out[k] = f * acc;
                    }
                }
            };
            // -0.5 |Ls^{-1} (v - m)|^2, the log proposal density up to the
            // constant that cancels in the ratio; forward substitution
            // against the lower-triangular factor.
            let log_q = |v: &[f64], m: &[f64], tmp: &mut [f64]| -> f64 {
                let mut q = 0.0f64;
                if proposal_l.is_empty() {
                    for j in 0..n_params {
                        let r = (v[j] - m[j]) / proposal_sd[pbase + j];
                        q -= 0.5 * r * r;
                    }
                } else {
                    for k in 0..n_params {
                        let mut acc = v[k] - m[k];
                        for j in 0..k {
                            acc -= proposal_l[lbase + k * n_params + j] * tmp[j];
                        }
                        tmp[k] = acc / proposal_l[lbase + k * n_params + k];
                        q -= 0.5 * tmp[k] * tmp[k];
                    }
                }
                q
            };

            for t in 0..n_iter {
                if mala {
                    // Metropolis-adjusted Langevin step. The drift pulls the
                    // proposal along the preconditioned gradient, and the
                    // acceptance carries the asymmetric-proposal correction
                    // q(x|y) - q(y|x) of the textbook MALA ratio.
                    half_m_times(&gx, &mut mean_x, &mut vbuf);
                    for j in 0..n_params {
                        mean_x[j] += state[j];
                        ctr += 1;
                        let u1 = rand_uniform(seedmix, ctr);
                        ctr += 1;
                        let u2 = rand_uniform(seedmix, ctr);
                        zbuf[j] = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                    }
                    if proposal_l.is_empty() {
                        for j in 0..n_params {
                            prop[j] = mean_x[j] + proposal_sd[pbase + j] * zbuf[j];
                        }
                    } else {
                        for k in 0..n_params {
                            let mut acc = mean_x[k];
                            for j in 0..=k {
                                acc += proposal_l[lbase + k * n_params + j]
                                    * zbuf[j];
                            }
                            prop[k] = acc;
                        }
                    }
                    let plp = lp_grad(&prop, &mut gy, &mut work);
                    half_m_times(&gy, &mut mean_y, &mut vbuf);
                    for j in 0..n_params {
                        mean_y[j] += prop[j];
                    }
                    let logq_diff = log_q(&state, &mean_y, &mut vbuf)
                        - log_q(&prop, &mean_x, &mut vbuf);
                    ctr += 1;
                    let u = rand_uniform(seedmix, ctr);
                    if plp.is_finite()
                        && u.ln() < (plp - cur) / temperature + logq_diff
                    {
                        state.copy_from_slice(&prop);
                        std::mem::swap(&mut gx, &mut gy);
                        cur = plp;
                        accepts += 1;
                    }
                    for j in 0..n_params {
                        cdraws[t * n_params + j] = state[j] as f32;
                    }
                    continue;
                }
                if proposal_mode == 0 {
                    // Gaussian random walk, diagonal scale.
                    for j in 0..n_params {
                        ctr += 1;
                        let u1 = rand_uniform(seedmix, ctr);
                        ctr += 1;
                        let u2 = rand_uniform(seedmix, ctr);
                        let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                        prop[j] = state[j] + proposal_sd[pbase + j] * z;
                    }
                } else if proposal_mode == 2 {
                    // Gaussian random walk with a per-chain lower-triangular
                    // Cholesky factor: prop = state + L z, so the proposal
                    // carries the full covariance the warmup estimated.
                    for j in 0..n_params {
                        ctr += 1;
                        let u1 = rand_uniform(seedmix, ctr);
                        ctr += 1;
                        let u2 = rand_uniform(seedmix, ctr);
                        zbuf[j] = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                    }
                    for k in 0..n_params {
                        let mut acc = state[k];
                        for j in 0..=k {
                            acc += proposal_l[lbase + k * n_params + j] * zbuf[j];
                        }
                        prop[k] = acc;
                    }
                } else {
                    // Differential Evolution: the proposal increment is the
                    // scaled difference of two other chains' batch-start
                    // states, read from the frozen `init` snapshot, plus a
                    // small per-dimension jitter. The pair is redrawn each
                    // iteration; with probability 0.1 the scale collapses to
                    // 1.0 for an occasional mode-crossing jump (ter Braak 2006).
                    ctr += 1;
                    let mut a = (rand_uniform(seedmix, ctr) * n_chains as f64)
                        as usize;
                    if a >= n_chains { a = n_chains - 1; }
                    ctr += 1;
                    let mut b = (rand_uniform(seedmix, ctr) * n_chains as f64)
                        as usize;
                    if b >= n_chains { b = n_chains - 1; }
                    if b == a { b = (b + 1) % n_chains; }
                    ctr += 1;
                    let g = if rand_uniform(seedmix, ctr) < 0.1 { 1.0 } else { gamma };
                    let abase = a * n_params;
                    let bbase = b * n_params;
                    for j in 0..n_params {
                        ctr += 1;
                        let u1 = rand_uniform(seedmix, ctr);
                        ctr += 1;
                        let u2 = rand_uniform(seedmix, ctr);
                        let z = (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos();
                        let diff = init[abase + j] - init[bbase + j];
                        prop[j] =
                            state[j] + g * diff + de_noise * proposal_sd[pbase + j] * z;
                    }
                }
                let plp = log_post(&prop);
                ctr += 1;
                let u = rand_uniform(seedmix, ctr);
                // Tempered acceptance: at temperature 1 this is the textbook
                // Metropolis ratio; at temperature T > 1 the chain accepts more
                // aggressively, which is what makes parallel tempering's hot
                // chains explore freely.
                if u.ln() < (plp - cur) / temperature {
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
            (cdraws, rate, cur)
        })
        .collect();

    // Assemble into the (n_iter, n_chains, n_params) column-major layout.
    let mut draws = vec![0.0f32; n_iter * n_chains * n_params];
    let mut accept_rate = vec![0.0f32; n_chains];
    let mut last_log_post = vec![0.0f64; n_chains];
    for (c, (cdraws, rate, lp)) in per_chain.iter().enumerate() {
        accept_rate[c] = *rate;
        last_log_post[c] = *lp;
        for t in 0..n_iter {
            for j in 0..n_params {
                draws[t + n_iter * (c + n_chains * j)] = cdraws[t * n_params + j];
            }
        }
    }

    ChainResult { n_iter, n_chains, n_params, draws, accept_rate, last_log_post }
}
