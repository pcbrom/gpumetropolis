//! Log-density kernels.
//!
//! The functions here are the swappable compute kernel of the package. Phase 0
//! evaluates them on the CPU; a later phase replaces the body with a CubeCL
//! kernel dispatched to the GPU, keeping the same signature.

/// Batched Gaussian-mean log-density kernel.
///
/// Target: the posterior of the mean `mu` of `data ~ Normal(mu, sigma^2)` under
/// a flat prior, which is proportional to
/// `-0.5 / sigma^2 * sum_i (data_i - mu)^2`. The normalising constant is
/// dropped because Metropolis-Hastings only needs the log-density up to an
/// additive constant.
///
/// `candidates` holds one `mu` proposal per chain (the chain axis); the inner
/// sum over `data` is the data-parallel axis. The two axes form the cartesian
/// product that the GPU kernel parallelises.
pub fn log_density_batch(candidates: &[f64], data: &[f64], sigma: f64) -> Vec<f64> {
    let inv_two_var = 0.5 / (sigma * sigma);
    candidates
        .iter()
        .map(|&mu| {
            let ss: f64 = data.iter().map(|&x| (x - mu) * (x - mu)).sum();
            -inv_two_var * ss
        })
        .collect()
}
