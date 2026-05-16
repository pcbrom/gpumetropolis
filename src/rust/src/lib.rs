use extendr_api::prelude::*;

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

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod gpumetropolis;
    fn rust_log_density_batch;
    fn rust_metropolis_gaussian_mean;
}
