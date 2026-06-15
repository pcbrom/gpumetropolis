# Experiment protocol: characterising the advantage regime of gpumetropolis

Version 0.6. Status: FROZEN. Pre-registered and approved by the author before
any cell of the registered experiment is executed. Freeze date of v0.1:
2026-05-16. Amendments v0.2 to v0.6: 2026-05-16. See section 14 for the
amendment record.

The protocol is versioned on the 0.x line because it is a living
pre-registration that is still being refined ahead of the registered run; a
1.x label would imply a final document.

Pre-registration in the strict sense: this document fixes the hypotheses,
design, metrics and decision rules before any result exists. The dated commit
that freezes it is the verifiable timestamp. No cell of the registered
experiment is run before the freeze.

## 1. Objective

Characterise, in a refutable way, the regime in which gpumetropolis beats,
ties, or loses to the established MCMC packages of the R ecosystem, along the
axes of speed, sampling stability and numerical stability. The objective is not
to demonstrate superiority; it is to map the advantage frontier with hypotheses
that are allowed to fail.

## 2. Binding caveats (inherited from the canonical plan)

These caveats constrain what the experiment may claim. Contradicting them
requires an explicit instruction from the author.

1. MCMC has an intrinsic sequential dependence; the time axis within a single
   chain does not parallelise.
2. The parallelism exploited comes from two axes: many independent chains, and
   the data-parallel evaluation of the log-density.
3. A GPU does not accelerate every MCMC. It pays off only for an expensive
   log-density (large N) or for many chains. For a small model with few chains
   the transfer overhead dominates and the GPU loses.
4. Low-level languages tie on peak performance. Rust's gain is not raw speed;
   no speedup is attributed to the choice of language.
5. MCMC equivalence is distributional, never a numerical epsilon. MCMC is
   stochastic; bit-exact reproducibility between CPU and GPU is not expected.

## 3. Pre-registered hypotheses

Each hypothesis carries a quantitative prediction, a support condition and a
refutation condition. Hypotheses H3 and H4 depend on the Phase 1 GPU kernel and
are tested only in the second execution stage (section 9).

| ID | Prediction | Supported when | Refuted when |
|---|---|---|---|
| H1 correctness | gpumetropolis and every competitor sample the same target posterior | across the family of all H1 KS tests of a backend, no rejection survives Holm-Bonferroni FWER control at family level `0.05` (see section 6.1, amended in v0.2); and the per-backend rate of split R-hat exceeding `1.01` is consistent with estimation noise | a rejection survives the correction, or R-hat shows systematic non-convergence; the defect is fixed before any speed is measured |
| H2 CPU parity | in the CPU regime, gpumetropolis-CPU does not dominate the competitors; the ratio of median ESS/s between gpumetropolis-CPU and the best competitor stays in `[0.5, 2]` for N up to `1e4` | the median ratio lies in `[0.5, 2]` with a bootstrap CI contained in `[0.33, 3]` | ratio `< 0.5` signals an inefficiency to fix; ratio `> 2` requires a harness audit, since it contradicts Caveat 4 |
| H3 GPU advantage regime | there exists an `N*` such that, for `N > N*`, the GPU backend of gpumetropolis beats the best CPU backend in ESS/s by a factor `>= 2`; the prediction places `N*` in the order `1e4` to `1e5` | some tested `N` up to `1e7` shows a median ratio `>= 2` with the lower CI bound `> 1` | no `N` up to `1e7` shows a median ratio `>= 2`: the value proposition fails and the scope is reassessed |
| H4 portability | the kernel passes distributional equivalence on NVIDIA and on AMD | the KS test does not reject at `alpha = 0.01` and R-hat `< 1.01` on both vendors | failure on any vendor beyond the development one |
| H5 numerical stability | for the Gaussian model with `N = 1e7`, the relative error of the log-density evaluated by the tree reduction grows as `O(log N)` in `epsilon`, against `O(N)` for the naive sequential sum | the tree-reduction error is smaller than the naive-sum error by `>= 1` order of magnitude, both measured against an extended-precision reference | the tree-reduction error is no better than the naive-sum error |
| H6 sampling stability | the failure rate of gpumetropolis (NaN, non-finite, R-hat `> 1.1`, divergence) is less than or equal to that of the median competitor across all cells | the failure rate is at or below the competitor median | the failure rate is above the competitor median |

H5 carries a recorded caveat from the outset: a well-implemented CPU sum
(pairwise, or compensated Kahan) reaches the same `O(log N)` bound or better.
The H5 comparison is between tree reduction and naive sum; the report also
states the tie against a well-implemented CPU sum. The CPU is not handicapped
to inflate the GPU advantage.

## 4. Factorial design

Crossed factorial design over the factors below. Each combination is a cell;
each cell receives the number of replications of section 7.

| Factor | Levels | Axis |
|---|---|---|
| N, data set size | `1e2, 1e3, 1e4, 1e5, 1e6, 1e7` | log-density cost (Caveat 3) |
| C, number of chains | `1, 8, 64, 512, 4096, 32768` | chain parallelism (Caveat 2) |
| backend | gpumetropolis-CPU; gpumetropolis-CUDA; gpumetropolis-Vulkan-NVIDIA; gpumetropolis-Vulkan-AMD; MCMCpack; mcmc; nimble; BayesianTools; greta; Stan via cmdstanr | implementation |
| target model | M1 to M4 (section 5) | target geometry |

Invalid cells are excluded and the exclusion is recorded: competitors without
native support for thousands of chains run up to the largest C they support
idiomatically; GPU backends exist only from Phase 1.

## 5. Target models

| ID | Model | Reference truth | Note |
|---|---|---|---|
| M1 | Gaussian mean, known `sigma`, flat prior | closed-form posterior `Normal(mean(data), sigma^2 / N)` | exact truth; already implemented |
| M2 | bimodal target (mixture of two separated Gaussians, or a bimodal Gumbel) | high-resolution numerical quadrature | tests mode crossing |
| M3 | heavy-tailed target (Student t with few degrees of freedom) | quadrature or a long reference sample | tests tail stability |
| M4 | high-curvature target (ill-conditioned Gaussian of moderate dimension) | closed-form when Gaussian | tests geometric robustness |

Execution order: M1 first, since it has an exact truth. M2 to M4 enter as the
harness matures.

Recorded scope boundary: the parameter dimension is kept low to moderate (1 to
about 10). The experiment claims nothing about high-dimensional targets.
Random-walk Metropolis scales poorly with dimension; this is known and is not
the package's claim. See section 11.

## 6. Response metrics

### 6.1 Correctness gate (precedes any speed comparison)

A cell enters the speed analysis only after passing the H1 gate. The speed of
an incorrect sampler is not measured.

The gate, as amended in v0.2, works as follows. For each replication of each
valid cell, a two-sample KS test compares the thinned pooled draws against an
exact sample from the reference truth. The draws are thinned to the effective
sample size before the test, because the KS test assumes independent draws.
This yields one p-value per replication, on the order of tens of thousands of
tests in stage A. The H1 verdict is not taken per test: that would be
unsatisfiable, since at any fixed per-test level a correct sampler still
produces spurious rejections at that rate. Instead the p-values of a backend
are treated as a family and the Holm-Bonferroni step-down procedure controls
the family-wise error rate at `0.05`. H1 is supported for that backend when no
rejection survives the procedure.

Recorded caveat: family-wise control makes the gate conservative as a
bug-detector. It has limited power against a small defect confined to one
corner of the factorial. To keep such a defect visible, the KS rejection rate
at the nominal per-test level is also reported per backend and per model as a
secondary diagnostic. The secondary diagnostic does not change the H1 verdict;
a rate well above nominal is a recorded signal for inspection even when no
single test survives the correction.

R-hat is recorded for every run as a convergence diagnostic. With the
iteration tuning of section 7 the registered runs target R-hat well below
1.01; the per-backend exceedance rate is reported alongside the KS family.

### 6.2 Primary efficiency metric

Effective Sample Size per second (ESS/s). ESS is computed uniformly for every
sampler with `coda::effectiveSize`, to remove the estimator as a confounder.
Time is the wall-clock of the sampling call, excluding setup and compilation.
The compilation time of nimble and Stan is reported separately as a one-time
cost.

### 6.3 Sampling stability

Distribution of R-hat across replications; variance of ESS; failure rate (NaN,
non-finite values, R-hat `> 1.1`, reported divergences); agreement under the
same seed on the same device.

### 6.4 Numerical stability

For M1 with large `N`: relative error of the posterior mean and variance
estimators against the closed-form truth; and relative error of the log-density
evaluation by the tree reduction and by the naive sum, both against an
extended-precision reference. The extended-precision reference is computed with
a correctly-rounded summation (Python `math.fsum`, or `mpmath` for the full
quantity).

## 7. Measurement protocol

- Replications per cell: 20, with distinct seeds (amended to 20 in v0.4; the
  frozen v0.1 value was 40, reduced to fit the 3 hour compute ceiling).
- Seed scheme: `seed = 10000 * cell_id + replication_index`, fixed in advance;
  the `cell_id` map is generated and frozen together with this document, in
  `benchmark/cell_map.csv`.
- Discard: warmup of half the iterations, following the package default;
  identical for every sampler.
- Iterations: tuned per model so the reference sampler reaches ESS `>= 400` per
  chain in the base regime; the same iteration budget is given to every
  competitor.
- Threads: the primary comparison runs single-threaded, to isolate the
  algorithm. A multi-thread variant is a secondary analysis, only where the
  package supports it.
- Hardware warmup: three discarded runs before each cell.
- Environment: versions of R, the Rust compiler, the packages, the GPU driver,
  the system kernel and the CPU frequency governor are recorded. The full host
  snapshot, machine and software, is captured in `benchmark/ENVIRONMENT.md` by
  the re-runnable script `benchmark/capture_env.sh`, so a third party can
  reproduce or audit the run.
- CPU baseline machine: AMD Ryzen 9 9900X3D, 24 threads. GPU backend machine:
  NVIDIA RTX 4090 and an AMD GPU on the same host.

### 7.1 Compute budget and the budget-exceeded outcome (amended in v0.3)

The factorial spans `N` up to `1e7` and `C` up to `32768`. The cost of one CPU
run is on the order of `n_iter * C * N`. The largest cells are infeasible on a
CPU by design: that infeasibility is the reason the package exists, and Stage B
runs those cells on the GPU. The CPU baseline is not required to complete them.

Each single run is therefore given a wall-clock budget `B`, fixed before the
run and recorded, default `B = 120` seconds, single-threaded. A run that
reaches `B` is terminated and recorded with outcome `budget-exceeded` and the
elapsed budget. A `budget-exceeded` outcome is a datum, not a failure: it
bounds the backend's ESS/s from above in that cell and, set against a Stage B
GPU cell that completes, is direct evidence for H3.

A cell verdict uses the replications that completed. A cell whose replications
all hit `B` is recorded as `intractable` for that backend. Runs are executed in
parallel across CPU cores; each run stays single-threaded, so parallelism
raises throughput without changing the per-run timing that the ESS/s metric
depends on.

The operational parameters of the registered Stage A M1 run are fixed by
amendment v0.4 to meet a 3 hour compute ceiling and a 30 GB memory ceiling set
by the author: `B = 30` s, 20 replications, 8 parallel workers, a virtual
memory cap per worker so the workers together cannot exceed the memory
ceiling, replication-major job ordering so an early stop leaves every cell with
an equal completed-replication count, and a global watchdog at 2 h 50 m that
guarantees the run ends within the ceiling.

## 8. Statistical analysis

Per cell, the median ESS/s and its 95 percent bootstrap confidence interval
over the completed replications are reported. The H2 and H3 comparisons use the ratio
of medians between gpumetropolis and the reference competitor, with a bootstrap
CI. The decision rules of section 3 are applied to those CIs, not to point
estimates. No decision rests on a difference of means without a CI.

## 9. Two-stage execution plan

- Stage A, now: CPU backend. Cells of gpumetropolis-CPU against MCMCpack, mcmc,
  nimble, BayesianTools, greta and Stan via cmdstanr, on model M1, then M2 to
  M4. Tests H1, H2, H5 and H6 on the CPU slice. Sets the honest baseline.
- Stage B, after Phase 1: GPU backends (CUDA, Vulkan-NVIDIA, Vulkan-AMD). Tests
  H3 and H4 and completes H5 and H6 on the GPU slice.

## 10. Decision criteria for CRAN publication

CRAN publication is blocked only by the failure of H1, that is, by
incorrectness. The failure of H3, the absence of a GPU advantage regime, does
not block publication; it forces a repositioning of the package's public
claims, and the final decision rests with the author. The package may go to
CRAN as a correct artefact occupying the empty niche even without a speed
dominance. The experiment characterises; it is not a dominance gatekeeper.

## 11. Threats to validity

- Distinct algorithms: the competitors use HMC, NUTS, DE-MCMC or Metropolis;
  comparing by ESS/s is algorithm-neutral, but the interpretation records that
  different samplers have different per-iteration costs.
- Low dimension favours random-walk Metropolis; the experiment does not
  generalise to high dimension and the report says so.
- Implementation maturity: the competitors have years of tuning; a
  gpumetropolis disadvantage may be one of implementation, not of algorithm.
- A backend whose log-density is an R callback (mcmc, MCMCpack) pays
  interpreter overhead per iteration, while a backend that compiles the density
  (gpumetropolis, nimble, Stan, greta) does not. A speed gap between these two
  groups is partly an implementation artefact, not an algorithmic result, and
  the interpretation records it.
- CPU-GPU transfer overhead varies with driver and bus; it is measured and
  reported, not assumed.
- ESS estimation is itself noisy; it is made uniform with `coda` and smoothed
  by the 40 replications.

## 12. What the experiment will not claim

- That gpumetropolis is the fastest MCMC. The defensible claim is conditional
  on the regime.
- That there is incontestable numerical stability. Numerical stability is
  reported as measured error against a high-precision reference, with
  rounding-error bounds cited from Higham (2002).
- That the gain comes from the Rust language. See Caveat 4.
- Anything about high-dimensional targets.

## 13. References

- Gelman, A. and Rubin, D. B. (1992). Inference from iterative simulation using
  multiple sequences. Statistical Science 7(4), 457-472.
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. Statistical Science
  7(4), 473-483.
- Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms, 2nd ed.
  SIAM.
- Plummer, M. et al. (2006). CODA: Convergence diagnosis and output analysis
  for MCMC. R News 6(1), 7-11.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  Scandinavian Journal of Statistics 6(2), 65-70.

## 14. Amendments

Amendments are disclosed here with date and rationale. The git history holds
the verbatim earlier version. An amendment is admissible only when it is made
before the registered run of the affected stage and is recorded transparently.

Note on numbering: the first three commits of this document labelled it 1.0,
1.1 and 1.2. It was renumbered to the 0.x line (0.1, 0.2, 0.3) on 2026-05-16,
since a 1.x label would imply a final document while the protocol is still
being refined ahead of the registered run. The earlier commit messages keep
their original labels; the mapping is 1.0 to 0.1, 1.1 to 0.2, 1.2 to 0.3.

### v0.2, 2026-05-16: H1 correctness criterion

Change. The v0.1 H1 criterion required that the KS test "does not reject at
`alpha = 0.01` ... in every valid cell". Section 3 and section 6.1 are amended
so the H1 verdict is taken on the family of KS tests of a backend, with
Holm-Bonferroni control of the family-wise error rate at `0.05`, plus a
secondary per-backend rejection-rate diagnostic.

Rationale. Stage A runs on the order of 40000 H1 KS tests (1008 cells times 40
replications). At a fixed per-test level a correct sampler still rejects at
that rate, so "no rejection in any cell" is not satisfiable by any sampler,
correct or not. The v0.1 criterion was therefore a logical defect, not a
discriminating test.

Discovery. The defect was found while validating the harness with a pilot. The
pilot is a pipeline check; it is not part of the registered experiment and its
numbers are not registered results. This amendment was made before the
registered stage A run began, so the pre-registration of the registered run is
intact.

Author decision. The author selected the multiple-testing-correction route
over a rejection-rate-equivalence route. Holm-Bonferroni is used rather than
plain Bonferroni because it controls the same family-wise error rate while
dominating Bonferroni in power.

### v0.3, 2026-05-16: compute budget per run

Change. Section 7.1 is added: each single run gets a wall-clock budget `B`
(default 120 s, single-threaded); a run reaching `B` is recorded as
`budget-exceeded`; a cell whose 40 replications all reach `B` is recorded as
`intractable` for that backend.

Rationale. The cost of one CPU run scales as `n_iter * C * N`. With the frozen
factorial, the largest cell costs on the order of `1e15` operations, hundreds
of days on one CPU core. The full Stage A factorial is therefore not
completable on a CPU, by arithmetic, not by choice of effort. This was known
in spirit from Caveat 3 but was not made operational in v0.1.

Why this is not a weakening of the design. The infeasible cells are precisely
the regime the GPU exists to serve. A `budget-exceeded` record is the honest
CPU datum for those cells and feeds H3 directly. Stage B runs the same cells
on the GPU. The amendment precedes the registered Stage A run.

### v0.5, 2026-05-16: first executed run is a reduced subset

Change. The author set a 20 to 30 minute wall-clock ceiling for the first
executed run. To fit it, the first run is a reduced subset, not the full
factorial of section 4:

- grid: N in `{1e3, 1e5, 1e7}` and C in `{1, 64, 4096}`, three levels each,
  spanning the small, medium and large regimes of both axes;
- model: M1 only;
- backends: gpumetropolis on cpu, cuda and vulkan, plus mcmc, MCMCpack,
  nimble, BayesianTools and Stan via cmdstanr (greta excluded as recorded;
  nimble excluded for N at or above `1e5` as recorded);
- replications: 10 per cell;
- per-run budget B: 20 s; global watchdog: 27 minutes.

Status of the full design. The full factorial of section 4, with 40
replications, remains the registered target. It is run later, without a tight
ceiling. The reduced run is labelled as such in every report.

Effect on the analysis. Ten replications on a three-by-three grid give wider
bootstrap confidence intervals and a coarser map of the advantage frontier
than the full design. The hypotheses and their decision rules are unchanged;
the verdicts from the reduced run are reported with that lower resolution
stated explicitly. The reduction is a loss of precision, not a change of the
questions. The amendment precedes the executed run.

### v0.6, 2026-05-16: one process per cell

Change. The execution unit becomes the cell, not the single run. One process
runs all ten replications of a cell. The per-run budget B is replaced by a
per-cell wall-clock cap (120 s); the global watchdog is unchanged.

Rationale. A first attempt with one process per replication showed every GPU
backend at zero completed cells: each fresh process pays CUDA or wgpu
initialisation and the CubeCL kernel JIT compilation, and that fixed cost
alone exceeds a 20 s per-run budget. That is a harness artefact, not a property
of the sampler. Batching the replications into one process, with a discarded
warmup call that absorbs the initialisation and the JIT, lets the timed
replications measure sampling only.

Effect on the metric. The metric is unchanged: `time_sec` still measures the
wall-clock of one sampling call, now with initialisation and JIT already paid
by the warmup. A cell whose process reaches the 120 s cap keeps the
replications that finished; the rest are recorded as budget-exceeded. The
amendment precedes the executed run; the earlier one-process-per-replication
attempt produced no registered results and is discarded.

### v0.4, 2026-05-16: 3 hour compute ceiling and 30 GB memory ceiling

Change. The author set a hard 3 hour wall-clock ceiling and a 30 GB memory
ceiling for the registered Stage A M1 run. Section 7 and section 7.1 are
amended: replications per cell from 40 to 20; per-run budget `B` from 120 s to
30 s; 8 parallel workers each under a virtual memory cap; replication-major job
ordering; a global watchdog at 2 h 50 m.

Rationale. The per-run budget alone does not bound the total: with the full
factorial a large fraction of cells reach `B`, and the sum still runs to tens
of hours. The author chose a fixed wall-clock ceiling instead. Replication-major
ordering means that if the watchdog stops the run early, every cell still has
the same number of completed replications, so the design stays balanced.

Effect on the analysis. Twenty replications still support a bootstrap
confidence interval for a median; the interval is wider than at 40. The
decision rules of section 3 are unchanged and are still applied to the
intervals. The reduction is recorded as a loss of precision, not a change of
the hypotheses. The amendment precedes the registered Stage A run.

### v0.7, 2026-05-18: the M2 to M4 extended run, recorded after the run

Status of this amendment. Amendments v0.2 to v0.6 were written before the run
they affect, as the rule of this section requires. This one is not: it
documents the M2 to M4 extended run after that run was executed, because the
M2 to M4 harness was built and run within a single working session. The
safeguard against a silent change is the git history, which holds the
verbatim model code, the harness and the commit timestamps, together with the
fact that the full factorial of section 4, twenty replications and no time
ceiling, remains the un-run registered target. The M2 to M4 run is an
extended run in the sense of v0.5, not the registered run, and is labelled as
such in every report.

The extended run. Models M2, M3 and M4 of section 5 were executed on
2026-05-18: M2 a separated bimodal posterior, M3 a heavy-tailed Student-t
location model, M4 an ill-conditioned three-dimensional Gaussian. Fifteen
replications per cell, a 60 s per-cell cap, a 70 minute global watchdog, four
parallel workers; 2568 replications completed.

Design choices recorded.

- M4 has no observed data; it is a fixed Gaussian target. The data-size axis N
  of the section-4 factorial does not apply to it, so M4 was swept over the
  chain count C only, at a single nominal N. The part of the section-4 grid
  that lists M4 against every N is recorded here as not applicable rather than
  run.
- The M2 reference truth is the closed-form symmetric bimodal posterior, not
  the numerical quadrature named in section 5. The closed form is exact up to
  a negligible truncation; the deviation is recorded for completeness.
- The M3 reference truth is one-dimensional quadrature of the log-posterior,
  which matches the section-5 description.
- Iteration budget: 4000 per chain for M2 and M3, 8000 for M4, which mixes
  slowly under the ill-conditioned geometry and is cheap per iteration.
- The chain count for M4 is capped at 4096: M4 is three-dimensional and its
  draw array grows as `n_iter * C * dim`, so `C = 32768` would reach tens of
  gigabytes of host memory.

Resource policy. The run is bounded to four parallel workers; a per-cell
memory guard records any cell whose draw array would exceed 2.5 GB as
budget-exceeded rather than running it; a RAM guardian aborts the run above a
memory threshold. This is the operational counterpart, for the M2 to M4 run,
of the v0.4 ceilings.

The random-number-stream correction. The M1 runs and this M2 to M4 run were
executed with a counter-based RNG that mixed the base seed additively with the
counter. The seed scheme of section 7, `10000 * cell_id + replication`,
assigns consecutive seeds to the replications of a cell, and consecutive seeds
under that RNG produced counter streams overlapping by a one-counter shift.
This correlated gpumetropolis's within-cell replications; the competitors,
seeded through R's Mersenne-Twister, were unaffected. The effect is on the
variance of gpumetropolis's per-cell KS rejection-rate estimate, not on its
expectation: each individual run is a correct, valid MCMC run, which a
separate long-chain convergence check confirms, a single chain of two million
iterations matching the exact posterior at every proposal scale. The package
was corrected after this run to hash the seed, so consecutive seeds give
independent streams. The ESS-per-second results and the H1 correctness
conclusions are unaffected; a re-run on the corrected RNG would tighten the
variance of the gpumetropolis rejection-rate estimates without changing a
verdict.

### v0.9, 2026-06-15: M2 focused re-run with parallel tempering

Status of this amendment. Recorded after the focused re-run of 2026-06-15.
The full registered M2 cell of section 4 stays the un-run target; this
amendment documents a single-cell extended run that probes one specific
question, whether `gpu_metropolis(method = "pt")` of the v0.3.0 release
recovers the bimodal posterior that random-walk Metropolis cannot.

The focused re-run. Three adapters were exercised on one M2 cell at
`N = 400` observations, `C = 8` chains, 4000 iterations per chain with
2000 discarded as warmup, twenty replications, seed scheme
`20260200 + 10000 * replication`. The adapters are
`gpumetropolis-cpu` (random-walk Metropolis, the M2 baseline of v0.7),
`gpumetropolis-cpu-pt` (parallel tempering through the v0.3.0
`method = "pt"` path with the default geometric ladder from 1 to 10 and
`swap_every = max(n_chains, 10)`) and `nimble` (the M2 strongest
competitor of the v0.7 run). The harness is at `benchmark/run_m2_pt.R`;
the per-adapter aggregate is versioned at
`benchmark/m2_pt_summary.csv`, and the per-replication CSV is generated
at `benchmark/results/m2_pt/m2_pt_20260615.csv`, kept outside the
repository in line with the v0.7 convention of versioning only the
cell-level summary.

Results, averaged over the twenty replications:

| adapter | KS p-value | R-hat | ESS | ESS per second | wall-clock (s) |
|---|---|---|---|---|---|
| gpumetropolis-cpu | 0.54 | 62.28 | 3525 | 27782 | 0.13 |
| gpumetropolis-cpu-pt | 0.18 | 1.00 | 671 | 3007 | 0.23 |
| nimble | 0.45 | 61.97 | 3652 | 26973 | 0.15 |

Reading. The M2 posterior is symmetric and bimodal; the natural
diagnostic of cross-mode mixing is the split R-hat, not the pooled KS
test. The pooled KS test sees a symmetric mixture of within-mode draws
that resembles the reference even when each chain is stuck in one mode,
which is why both `gpumetropolis-cpu` and `nimble` look passable on KS.
Their R-hat of 62 is the honest report: the chains never mix between
modes, and every per-chain estimate is biased. Only `gpumetropolis-cpu-pt`
gets to R-hat near 1, because the swap step shuttles states between the
cold chain and the hot chains. Its ESS of 671 is the honest count: each
post-warmup draw of the cold chain is actually from the joint bimodal
posterior, paying autocorrelation across mode-crossing for that
correctness.

The nominal ESS per second of the random-walk adapters, 27782 against
3007 for parallel tempering, compares the throughput of stuck chains to
that of a mixing chain. The honest comparison conditions on
correctness: at R-hat 62 the effective sample size of valid posterior
draws is zero, so the random-walk adapters' valid ESS per second is
zero. Parallel tempering pays a 1.7x wall-clock factor over the
random-walk baseline (0.23 s against 0.13 s, the cost of the host-side
swap step between batches) and that buys the R-hat-near-1 verdict.

Mode coverage. The per-replication mode coverage column in the CSV
is `NA`: the script records adapter-level metrics rather than the raw
draws across replications. A future revision of the harness will store
draws per replication so per-replication mode coverage can be
recomputed; the M2 v0.7 run already provides the within-chain mode
counts that, with the v0.3.0 R-hat numbers above, document the
mode-crossing improvement.

Design choices recorded for this focused re-run.

- The cell is intentionally narrow, one `N`, one `C`, twenty
  replications. The full registered factorial of section 4, twenty
  replications per cell across all `N` and `C`, remains the un-run
  target; this re-run is not a substitute.
- The competitor set is reduced to `nimble`, the strongest M2
  competitor of v0.7. `MCMCpack`, `mcmc`, `BayesianTools` and Stan are
  omitted because their v0.7 vereditos on this cell do not change with
  the addition of parallel tempering on the gpumetropolis side.
- The proposal scale is `2.4 * sigma / sqrt(N)`, the M2 default of
  `m2_bimodal.R`. Per-chain adaptation in the v0.3.0 path refines it
  from that seed; the warmup acceptance per chain is recorded in
  `fit$adaptation$accept_history`.
- Parallel tempering activates `adapt = TRUE` by default; the cold
  chain inherits a proposal scale tuned to its tempered geometry, and
  the hot chains inherit larger ones, as the v0.3.0 release notes
  describe.

What this amendment does not change. The full M2 to M4 factorial of v0.7
is not re-run. The v0.7 vereditos on M3 and M4 stand. The v0.3.0
release notes explain the parallel-tempering path; this amendment
records the focused empirical evidence on the one cell where it
matters most.
