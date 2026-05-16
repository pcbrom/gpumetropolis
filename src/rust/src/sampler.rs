//! Batched random-walk Metropolis sampler (CPU reference).

use rand::Rng;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;

use crate::model::log_density_batch;

/// Result of a sampler run.
///
/// `draws` is stored column-major: the value of chain `c` at iteration `r`
/// lives at `draws[c * n_iter + r]`.
pub struct SamplerOutput {
    pub n_iter: usize,
    pub n_chains: usize,
    pub draws: Vec<f64>,
    pub accept_rate: Vec<f64>,
}

/// Run the batched random-walk Metropolis sampler for the Gaussian-mean model.
///
/// One independent chain is run per entry of `init`. Each chain owns a PCG64
/// stream seeded from `(seed, chain_index)`, so the output is reproducible and
/// does not depend on how the chains are scheduled. Every iteration gathers the
/// proposals of all chains and evaluates them with a single batched kernel
/// call: this orchestrator pattern is what a later phase keeps unchanged when
/// the kernel moves to the GPU. The sequential dependence inside each chain
/// (`x_{t+1}` depends on `x_t`) stays on the CPU, since it cannot be
/// parallelised; the parallel axes are the chains and the data sum.
pub fn metropolis_gaussian_mean(
    data: &[f64],
    sigma: f64,
    n_iter: usize,
    init: &[f64],
    proposal_sd: f64,
    seed: u64,
) -> SamplerOutput {
    let n_chains = init.len();
    let mut rng: Vec<Pcg64> = (0..n_chains)
        .map(|c| Pcg64::new(seed as u128, c as u128))
        .collect();

    let mut states: Vec<f64> = init.to_vec();
    let mut current_lp = log_density_batch(&states, data, sigma);

    let mut draws = vec![0.0_f64; n_iter * n_chains];
    let mut accepts = vec![0_u64; n_chains];
    let mut proposals = vec![0.0_f64; n_chains];

    for t in 0..n_iter {
        for c in 0..n_chains {
            let z: f64 = rng[c].sample(StandardNormal);
            proposals[c] = states[c] + proposal_sd * z;
        }
        let prop_lp = log_density_batch(&proposals, data, sigma);
        for c in 0..n_chains {
            let log_u = rng[c].gen::<f64>().ln();
            if log_u < prop_lp[c] - current_lp[c] {
                states[c] = proposals[c];
                current_lp[c] = prop_lp[c];
                accepts[c] += 1;
            }
            draws[c * n_iter + t] = states[c];
        }
    }

    let accept_rate = accepts
        .iter()
        .map(|&a| if n_iter == 0 { 0.0 } else { a as f64 / n_iter as f64 })
        .collect();

    SamplerOutput {
        n_iter,
        n_chains,
        draws,
        accept_rate,
    }
}
