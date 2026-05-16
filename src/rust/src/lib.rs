use extendr_api::prelude::*;

mod gpu;
mod interp;
mod model;
mod sampler;

/// Evaluate the batched Gaussian-mean log-density kernel.
///
/// Internal helper that exposes the compute kernel in isolation so it can be
/// checked directly. `candidates` holds one mean proposal per chain; the return
/// value holds one log-density per chain.
/// @noRd
#[extendr]
fn rust_log_density_batch(candidates: Vec<f64>, data: Vec<f64>, sigma: f64) -> Vec<f64> {
    model::log_density_batch(&candidates, &data, sigma)
}

/// Run the batched random-walk Metropolis sampler (CPU reference).
///
/// Internal worker behind the R function `metropolis_gaussian_mean()`. Returns
/// a list with the `n_iter` by `n_chains` matrix of draws and the per-chain
/// acceptance rate.
/// @noRd
#[extendr]
fn rust_metropolis_gaussian_mean(
    data: Vec<f64>,
    sigma: f64,
    n_iter: i32,
    init: Vec<f64>,
    proposal_sd: f64,
    seed: f64,
) -> List {
    let res = sampler::metropolis_gaussian_mean(
        &data,
        sigma,
        n_iter as usize,
        &init,
        proposal_sd,
        seed as u64,
    );
    let n_iter = res.n_iter;
    let draws = res.draws.clone();
    let mat = RArray::new_matrix(n_iter, res.n_chains, |r, c| draws[c * n_iter + r]);
    list!(draws = mat, accept_rate = res.accept_rate)
}

/// Evaluate the batched Gaussian-mean log-density through the CubeCL kernel.
///
/// Internal helper. `backend` selects the compute runtime: "cpu" or "cuda".
/// Used to check that the CubeCL path is wired into the package build.
/// @noRd
#[extendr]
fn rust_gaussian_logdens_gpu(
    candidates: Vec<f64>,
    data: Vec<f64>,
    sigma: f64,
    backend: &str,
) -> Vec<f64> {
    let cand: Vec<f32> = candidates.iter().map(|&v| v as f32).collect();
    let dat: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    gpu::gaussian_logdens(&cand, &dat, sigma as f32, gpu::Backend::from_name(backend))
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

/// Evaluate a compiled log-likelihood bytecode program over all chains.
///
/// Internal worker behind the generic API. `code` holds opcode/arg pairs,
/// `params` holds `n_chains * n_params` values grouped by chain. Returns one
/// log-likelihood sum per chain.
/// @noRd
#[extendr]
fn rust_loglik_sum(
    code: Vec<i32>,
    consts: Vec<f64>,
    n_params: i32,
    data: Vec<f64>,
    n_cols: i32,
    n_obs: i32,
    params: Vec<f64>,
    n_chains: i32,
    backend: &str,
) -> Vec<f64> {
    let code_u: Vec<u32> = code.iter().map(|&v| v as u32).collect();
    let consts_f: Vec<f32> = consts.iter().map(|&v| v as f32).collect();
    let data_f: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let params_f: Vec<f32> = params.iter().map(|&v| v as f32).collect();
    let prog = interp::Program {
        code: &code_u,
        consts: &consts_f,
        n_params: n_params as u32,
        data: &data_f,
        n_cols: n_cols as u32,
        n_obs: n_obs as u32,
    };
    interp::loglik_sum(&prog, &params_f, n_chains as usize, gpu::Backend::from_name(backend))
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

/// Run the generic batched Metropolis sampler over a compiled model.
///
/// Internal worker behind `gpu_metropolis()`. The log-density is given as
/// compiled bytecode; `draws` is returned flat in the column-major order of an
/// R array of dimension (n_iter, n_chains, n_params).
/// @noRd
#[extendr]
fn rust_gpu_metropolis(
    loglik_code: Vec<i32>,
    loglik_consts: Vec<f64>,
    n_params: i32,
    data: Vec<f64>,
    n_cols: i32,
    n_obs: i32,
    prior_code: Vec<i32>,
    prior_consts: Vec<f64>,
    init: Vec<f64>,
    proposal_sd: Vec<f64>,
    n_iter: i32,
    seed: f64,
    backend: &str,
) -> List {
    let ll_code: Vec<u32> = loglik_code.iter().map(|&v| v as u32).collect();
    let ll_consts: Vec<f32> = loglik_consts.iter().map(|&v| v as f32).collect();
    let data_f: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let prog = interp::Program {
        code: &ll_code,
        consts: &ll_consts,
        n_params: n_params as u32,
        data: &data_f,
        n_cols: n_cols as u32,
        n_obs: n_obs as u32,
    };
    let pr_code: Vec<u32> = prior_code.iter().map(|&v| v as u32).collect();
    let pr_consts: Vec<f32> = prior_consts.iter().map(|&v| v as f32).collect();
    let init_f: Vec<f32> = init.iter().map(|&v| v as f32).collect();
    let psd_f: Vec<f32> = proposal_sd.iter().map(|&v| v as f32).collect();

    let res = interp::gpu_metropolis(
        &prog,
        &pr_code,
        &pr_consts,
        &init_f,
        &psd_f,
        n_iter as usize,
        seed as u64,
        gpu::Backend::from_name(backend),
    );
    let draws: Vec<f64> = res.draws.iter().map(|&v| v as f64).collect();
    let accept_rate: Vec<f64> = res.accept_rate.iter().map(|&v| v as f64).collect();
    list!(
        draws = draws,
        accept_rate = accept_rate,
        n_iter = res.n_iter as i32,
        n_chains = res.n_chains as i32,
        n_params = res.n_params as i32
    )
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod gpumetropolis;
    fn rust_log_density_batch;
    fn rust_metropolis_gaussian_mean;
    fn rust_gaussian_logdens_gpu;
    fn rust_loglik_sum;
    fn rust_gpu_metropolis;
}
