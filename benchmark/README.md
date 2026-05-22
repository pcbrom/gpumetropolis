# benchmark/

Harness for the pre-registered experiment of `../EXPERIMENT_PROTOCOL.md`. This
directory is excluded from the package tarball through `.Rbuildignore`.

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

## Resource policy

A benchmark run sweeps cells whose draw array grows as `n_iter * C * dim`,
which at the largest chain counts reaches tens of gigabytes and can exhaust
host memory. Every run on a shared workstation follows this policy, and any
new run script must keep to it.

- **Parallelism capped at 4 cores.** Orchestrators set `parallel_jobs <- 4L`,
  leaving the rest of the machine responsive. A higher count is allowed only
  on a dedicated host.
- **Per-cell memory guard.** `run_cell_v2.R` estimates `n_iter * C * dim * 8`
  bytes before running a cell and, above a 2.5 GB cap, records the cell as
  budget-exceeded instead of running it. An oversized cell cannot crash the
  host; it is reported, not executed.
- **Bounded grid.** A model whose draw array is large at high `C`, such as the
  three-dimensional M4, has its chain count capped in the orchestrator so no
  cell approaches the guard in normal operation.
- **RAM guardian.** A long run is shadowed by a watcher that aborts the whole
  run if host memory use crosses a safe threshold, so a single unforeseen cell
  cannot take the system down.
- **Global watchdog.** Every orchestrator wraps the run in a wall-clock
  watchdog; cells not reached are recorded as budget-exceeded, which the
  protocol's budget mechanism already accounts for.
