use extendr_api::prelude::*;

mod chain_kernel;
mod cpu_native;
mod gpu;
mod jit;
mod model;
mod sampler;

/// Evaluate the batched Gaussian-mean log-density kernel.
///
/// Internal helper behind the Phase 0 reference sampler.
/// @noRd
#[extendr]
fn rust_log_density_batch(candidates: Vec<f64>, data: Vec<f64>, sigma: f64) -> Vec<f64> {
    model::log_density_batch(&candidates, &data, sigma)
}

/// Run the batched random-walk Metropolis sampler (CPU reference).
///
/// Internal worker behind the R function `metropolis_gaussian_mean()`.
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

/// Names of the compute backends compiled into this build.
///
/// Always includes "cpu"; "cuda" and "vulkan" are present only when the
/// matching Cargo feature was enabled at build time.
/// @noRd
#[extendr]
fn rust_available_backends() -> Vec<String> {
    let mut v = vec!["cpu".to_string()];
    if cfg!(feature = "cuda") {
        v.push("cuda".to_string());
    }
    if cfg!(feature = "vulkan") {
        v.push("vulkan".to_string());
    }
    v
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
    temperatures: Vec<f64>,
    n_iter: i32,
    seed: f64,
    backend: &str,
    proposal_mode: i32,
    gamma: f64,
    de_noise: f64,
    proposal_l: Vec<f64>,
) -> List {
    let ll_code: Vec<u32> = loglik_code.iter().map(|&v| v as u32).collect();
    let pr_code: Vec<u32> = prior_code.iter().map(|&v| v as u32).collect();

    let res = chain_kernel::gpu_metropolis_chain(
        &ll_code,
        &loglik_consts,
        &pr_code,
        &prior_consts,
        n_params as usize,
        &data,
        n_cols as usize,
        n_obs as usize,
        &init,
        &proposal_sd,
        &temperatures,
        n_iter as usize,
        seed as u32,
        gpu::Backend::from_name(backend),
        proposal_mode as u32,
        gamma,
        de_noise,
        &proposal_l,
    );
    let draws: Vec<f64> = res.draws.iter().map(|&v| v as f64).collect();
    let accept_rate: Vec<f64> = res.accept_rate.iter().map(|&v| v as f64).collect();
    list!(
        draws = draws,
        accept_rate = accept_rate,
        last_log_post = res.last_log_post,
        n_iter = res.n_iter as i32,
        n_chains = res.n_chains as i32,
        n_params = res.n_params as i32
    )
}

/// Evaluate the compiled log-likelihood at a batch of parameter points.
///
/// Internal worker behind the `gpum_crlb()` observed-information diagnostic.
/// `points` is the flat point-major buffer of `n_points` by `n_params`; the
/// return is the log-likelihood, summed over observations, at each point.
/// @noRd
#[extendr]
fn rust_loglik_batch(
    loglik_code: Vec<i32>,
    loglik_consts: Vec<f64>,
    n_params: i32,
    data: Vec<f64>,
    n_cols: i32,
    n_obs: i32,
    points: Vec<f64>,
) -> Vec<f64> {
    let code: Vec<u32> = loglik_code.iter().map(|&v| v as u32).collect();
    cpu_native::loglik_batch(
        &code,
        &loglik_consts,
        n_params as usize,
        &data,
        n_cols as usize,
        n_obs as usize,
        &points,
    )
}

/// Gradient of the compiled log-likelihood (data term, summed over the
/// observations) at a batch of points, by reverse-mode automatic
/// differentiation of the bytecode. Point-major flat output of length
/// `n_points * n_params`.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rust_grad_batch(
    loglik_code: Vec<i32>,
    loglik_consts: Vec<f64>,
    n_params: i32,
    data: Vec<f64>,
    n_cols: i32,
    n_obs: i32,
    points: Vec<f64>,
) -> Vec<f64> {
    let code: Vec<u32> = loglik_code.iter().map(|&v| v as u32).collect();
    cpu_native::grad_batch(
        &code,
        &loglik_consts,
        n_params as usize,
        &data,
        n_cols as usize,
        n_obs as usize,
        &points,
    )
}

/// Evaluate the compiled log-likelihood per observation at a batch of points.
///
/// Internal worker behind `gpum_waic()` and `gpum_loo()`. Returns the flat
/// `n_points` by `n_obs` matrix in point-major order: entry `p * n_obs + i` is
/// the log-likelihood of observation `i` at point `p`.
/// @noRd
#[extendr]
fn rust_loglik_pointwise(
    loglik_code: Vec<i32>,
    loglik_consts: Vec<f64>,
    n_params: i32,
    data: Vec<f64>,
    n_cols: i32,
    n_obs: i32,
    points: Vec<f64>,
) -> Vec<f64> {
    let code: Vec<u32> = loglik_code.iter().map(|&v| v as u32).collect();
    cpu_native::loglik_pointwise(
        &code,
        &loglik_consts,
        n_params as usize,
        &data,
        n_cols as usize,
        n_obs as usize,
        &points,
    )
}

/// Run the synchronous Differential Evolution sampler (path B, CPU native).
///
/// Internal worker behind `gpu_metropolis(method = "de", de_sync = TRUE)`.
/// The population advances one generation at a time behind a barrier with a
/// double buffer, the canonical per-generation DE-MC mixing.
/// @noRd
#[extendr]
fn rust_gpu_metropolis_de_sync(
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
    gamma: f64,
    de_noise: f64,
) -> List {
    let ll_code: Vec<u32> = loglik_code.iter().map(|&v| v as u32).collect();
    let pr_code: Vec<u32> = prior_code.iter().map(|&v| v as u32).collect();
    let res = cpu_native::run_de_sync(
        &ll_code,
        &loglik_consts,
        &pr_code,
        &prior_consts,
        n_params as usize,
        &data,
        n_cols as usize,
        n_obs as usize,
        &init,
        &proposal_sd,
        n_iter as usize,
        seed as u32,
        gamma,
        de_noise,
    );
    let draws: Vec<f64> = res.draws.iter().map(|&v| v as f64).collect();
    let accept_rate: Vec<f64> = res.accept_rate.iter().map(|&v| v as f64).collect();
    list!(
        draws = draws,
        accept_rate = accept_rate,
        last_log_post = res.last_log_post,
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
    fn rust_available_backends;
    fn rust_gpu_metropolis;
    fn rust_loglik_batch;
    fn rust_loglik_pointwise;
    fn rust_gpu_metropolis_de_sync;
    fn rust_grad_batch;
}
