# benchmark/

Harness for the pre-registered experiment of `../EXPERIMENT_PROTOCOL.md`. This
directory is excluded from the CRAN tarball through `.Rbuildignore`.

## Frozen pre-registration artefacts

These files are committed and must not change once results exist. They are the
verifiable record that the design preceded the data.

- `generate_cell_map.R`: enumerates the factorial design and assigns the stable
  `cell_id` used by the seed scheme.
- `cell_map.csv`: the frozen cell map, 1440 cells (1008 stage A, 432 stage B).

## Layout (as the harness is built)

- `models/`: target model specifications (M1 to M4) with their reference truth.
- `adapters/`: one adapter per backend, exposing a uniform interface so every
  sampler is driven and measured the same way.
- `run_stage_a.R`: driver for the CPU stage.
- `analyze.R`: aggregation, bootstrap confidence intervals and hypothesis
  verdicts.
- `results/`: raw and aggregated run output. Gitignored; regenerable.

## Adapter contract

Every adapter is a function `adapter_<backend>(model, data, n_iter, n_chains,
seed)` returning a list with `draws` (iterations by chains matrix or array),
`time_sec` (wall-clock of the sampling call only), and `meta` (versions,
compile time, divergences). ESS is computed downstream with
`coda::effectiveSize`, uniformly, so the estimator is not a confounder.
