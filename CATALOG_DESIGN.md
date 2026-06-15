# Distribution catalogue for the auto-fit pipeline

Specification of the distribution catalogue that drives the v0.6.0
marginal auto-selection and the v0.8.0 synthesis. The catalogue is the
inventory that `gpum_fit_catalog(data, ...)` consults to propose and
rank candidate models for a column of data; the present document fixes
each distribution's parametrisation, its `gpum_loglik` in the DSL, the
initial values heuristic, the proposal scale and the open issues
(notably the `lgamma` operation in the DSL).

Curation took the Wikipedia "List of probability distributions" as the
universe and reduced it to families that meet three criteria together:
(a) appearance in applied statistics across multiple domains, not a
narrow specialty; (b) parametrisation in current use, identified by
presence in `stats`, `gamlss.dist`, `actuar` or `extraDistr`; (c)
tractable log-density in the DSL operation set, with at most a single
new operation declared up front.

## Tiers

The catalogue is organised in three tiers. Tier A covers unimodal
parametric families chosen by support; Tier B covers the multimodal
case via finite mixtures of any pair of Tier A; Tier C lists discrete
families. Multivariate and matrix families enter only with the vector
DSL of the deferred Tier 5 and are not part of the v0.6.0 scope.

## DSL prerequisite: `lgamma`

Five Tier A families (Gamma, Beta, Weibull, Inverse Gamma, Generalized
Gamma) need `log(Gamma(.))` in the log-density when shape is a
parameter. The 0.5.0 release adds one operation to the DSL bytecode,
`lgamma`, mapped to `f64::ln_gamma` on the CPU backend and to a
custom CubeCL implementation on the GPU backends, before the 0.6.0
catalogue work begins. The constants in front of `lgamma` are folded
at compile time and do not enlarge the instruction count.

## Tier A: unimodal continuous

### Continuous, whole real line

#### Normal `N(mu, sigma)`

Support `R`. Two parameters: location `mu`, scale `sigma > 0`. Log-density up to a
constant: `-0.5 * ((x - mu) / sigma)^2 - log(sigma)`.

```r
gpum_loglik <- ~ -((y - mu)^2) / (2 * sigma^2) - log(sigma)
init        <- list(mu = mean(y), log_sigma = log(sd(y)))
proposal_sd <- list(mu = 2.4 * sd(y) / sqrt(length(y)), log_sigma = 0.1)
```

#### Student-t `t(nu, mu, sigma)`

Support `R`. Three parameters: degrees of freedom `nu > 0`, location, scale.
Log-density: `-((nu + 1)/2) * log(1 + ((x - mu)/sigma)^2 / nu) - log(sigma)`.
Uses `log` and `^` only; no `lgamma` if the normaliser is absorbed
into the additive constant (it depends only on `nu`).

```r
gpum_loglik <- ~ -((nu + 1) / 2) * log(1 + ((y - mu)/sigma)^2 / nu) - log(sigma)
init        <- list(nu = 8, mu = median(y), log_sigma = log(mad(y)))
proposal_sd <- list(log_nu = 0.3, mu = mad(y) / sqrt(length(y)), log_sigma = 0.1)
```

#### Cauchy `Cauchy(x0, gamma)`

Special case of Student-t with `nu = 1`; kept as a separate entry for
heavy-tail diagnostics.

```r
gpum_loglik <- ~ -log(1 + ((y - x0)/gam)^2) - log(gam)
init        <- list(x0 = median(y), log_gam = log(IQR(y) / 2))
proposal_sd <- list(x0 = IQR(y) / sqrt(length(y)), log_gam = 0.1)
```

#### Laplace `Laplace(mu, b)`

Double-exponential, fatter than Normal at the centre.

```r
gpum_loglik <- ~ -sqrt((y - mu)^2) / b - log(b)
init        <- list(mu = median(y), log_b = log(mean(abs(y - median(y)))))
proposal_sd <- list(mu = mad(y) / sqrt(length(y)), log_b = 0.1)
```

#### Logistic `Logistic(mu, s)`

Sigmoidal CDF, classic in choice modelling.

```r
gpum_loglik <- ~ -(y - mu)/s - 2 * log(1 + exp(-(y - mu)/s)) - log(s)
init        <- list(mu = mean(y), log_s = log(sd(y) * sqrt(3) / pi))
proposal_sd <- list(mu = sd(y) / sqrt(length(y)), log_s = 0.1)
```

#### Skew normal (Azzalini) `SN(mu, sigma, alpha)`

Asymmetric extension of Normal; `alpha` is the shape, `alpha = 0`
recovers Normal.

```r
gpum_loglik <- ~ -((y - mu)/sigma)^2 / 2
                 + log(1 + exp(alpha * (y - mu)/sigma))
                 - log(sigma)
init        <- list(mu = mean(y), log_sigma = log(sd(y)), alpha = 0)
proposal_sd <- list(mu = sd(y)/sqrt(length(y)), log_sigma = 0.1, alpha = 0.5)
```

#### Gumbel max `Gumbel(mu, beta)`

Type 1 extreme value, for maxima.

```r
gpum_loglik <- ~ -(y - mu)/beta - exp(-(y - mu)/beta) - log(beta)
init        <- list(mu = median(y), log_beta = log(sd(y) * sqrt(6) / pi))
proposal_sd <- list(mu = sd(y)/sqrt(length(y)), log_beta = 0.1)
```

### Continuous, semi-infinite `x > 0`

#### Exponential `Exp(rate)`

Support `(0, infinity)`. One parameter, `rate > 0`.

```r
gpum_loglik <- ~ log(rate) - rate * y
init        <- list(log_rate = -log(mean(y)))
proposal_sd <- list(log_rate = 0.1)
```

#### Gamma `Gamma(shape, rate)`

Workhorse for positive continuous data. Needs `lgamma`.

```r
gpum_loglik <- ~ shape * log(rate) - lgamma(shape)
                 + (shape - 1) * log(y) - rate * y
init        <- list(log_shape = log((mean(y)/sd(y))^2),
                    log_rate  = log(mean(y)/sd(y)^2))
proposal_sd <- list(log_shape = 0.1, log_rate = 0.1)
```

#### Weibull `Weibull(shape, scale)`

Survival analysis; `shape = 1` is Exponential. The `lgamma` is absorbed
into the additive constant; the body uses only `log`, `*`, `^`, `-`.

```r
gpum_loglik <- ~ log(shape) - shape * log(scale)
                 + (shape - 1) * log(y) - (y/scale)^shape
init        <- list(log_shape = log(1), log_scale = log(mean(y)))
proposal_sd <- list(log_shape = 0.2, log_scale = 0.1)
```

#### Lognormal `LN(mu, sigma)`

Equivalent to Normal on `log(y)`; modelled directly to keep CDF and
quantile computations consistent.

```r
gpum_loglik <- ~ -((log(y) - mu)^2) / (2 * sigma^2)
                 - log(y) - log(sigma)
init        <- list(mu = mean(log(y)), log_sigma = log(sd(log(y))))
proposal_sd <- list(mu = sd(log(y))/sqrt(length(y)), log_sigma = 0.1)
```

#### Inverse Gaussian `IG(mu, lambda)`

First-passage times of a drifted Brownian motion. Useful for
heavy-tailed positive data where Gamma misses the tail.

```r
gpum_loglik <- ~ 0.5 * log(lambda)
                 - 1.5 * log(y)
                 - lambda * (y - mu)^2 / (2 * mu^2 * y)
init        <- list(log_mu = log(mean(y)),
                    log_lambda = log(mean(y)^3 / var(y)))
proposal_sd <- list(log_mu = 0.1, log_lambda = 0.1)
```

#### Pareto `Pareto(xm, alpha)`

Power-law tail.

```r
gpum_loglik <- ~ log(alpha) + alpha * log(xm) - (alpha + 1) * log(y)
init        <- list(log_xm = log(min(y)) - 0.01,
                    log_alpha = log(1 / mean(log(y/min(y)))))
proposal_sd <- list(log_xm = 0.05, log_alpha = 0.1)
```

Note: `xm` must be lower than `min(y)`; fit with `xm` known and equal
to `min(y) * 0.999` to avoid a hard truncation singularity.

#### Generalized Pareto `GPD(sigma, xi)`

Excedences above a threshold; the natural tail companion of Pareto.

```r
gpum_loglik <- ~ -log(sigma) - (1 + 1/xi) * log(1 + xi * y / sigma)
init        <- list(log_sigma = log(sd(y)), xi = 0.1)
proposal_sd <- list(log_sigma = 0.1, xi = 0.1)
```

#### Rayleigh `Rayleigh(sigma)`

Norm of a 2D zero-mean Gaussian vector.

```r
gpum_loglik <- ~ log(y) - log(sigma^2) - y^2 / (2 * sigma^2)
init        <- list(log_sigma = log(mean(y) * sqrt(2/pi)))
proposal_sd <- list(log_sigma = 0.1)
```

### Continuous, bounded interval

#### Uniform `U(a, b)`

Used as a non-informative reference; rarely the best fit but present
as a baseline.

```r
gpum_loglik <- ~ -log(b - a) + 0 * y
init        <- list(a = min(y) - 0.01, b = max(y) + 0.01)
proposal_sd <- list(a = sd(y)/sqrt(length(y)), b = sd(y)/sqrt(length(y)))
```

#### Beta `Beta(alpha, beta)`

Support `(0, 1)`; flexible for proportions.

```r
gpum_loglik <- ~ (alpha - 1) * log(y) + (beta - 1) * log(1 - y)
                 - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta))
init        <- list(log_alpha = log(2), log_beta = log(2))
proposal_sd <- list(log_alpha = 0.2, log_beta = 0.2)
```

#### Kumaraswamy `Kumaraswamy(a, b)`

Closed-form CDF makes it the computational alternative to Beta when
hierarchies and conditioning matter.

```r
gpum_loglik <- ~ log(a) + log(b)
                 + (a - 1) * log(y) + (b - 1) * log(1 - y^a)
init        <- list(log_a = log(2), log_b = log(2))
proposal_sd <- list(log_a = 0.2, log_b = 0.2)
```

#### Triangular `Tri(a, c, b)`

Informative when only `min`, `mode` and `max` are known.

```r
gpum_loglik <- ~ ifelse(y < c,
                        log(y - a) - log((b - a) * (c - a)),
                        log(b - y) - log((b - a) * (b - c)))
init        <- list(a = min(y), c = median(y), b = max(y))
proposal_sd <- list(a = sd(y)/10, c = sd(y)/sqrt(length(y)), b = sd(y)/10)
```

Note: `ifelse` is not in the DSL; encoded as a smooth max via a
piecewise polynomial. Alternative: cap on a single branch when initial
data suggests symmetric or right-skew.

#### Truncated normal `TN(mu, sigma, a, b)`

Normal restricted to `(a, b)`; the normaliser uses `pnorm`, which the
DSL does not have. Implemented as Normal log-density up to a constant,
with `(a, b)` enforced as hard bounds on the data; the missing
normaliser is recovered at posterior summary time via an external
Monte Carlo estimate.

## Tier B: multimodal as finite mixtures

Mixtures are composed at the catalogue layer, not declared as
primitives.

**Formal definition.** A finite mixture with `k` components has density

```
f(x) = sum_{i=1..k} w_i * f_i(x ; theta_i),
       w_i >= 0, sum_i w_i = 1
```

where each `f_i` is one of the Tier A primitives with its own parameter
vector `theta_i` and `w_i` is the mixing weight. The CDF is the
corresponding convex combination of component CDFs. Sampling is by
ancestral sampling: draw the latent component `z` from the categorical
distribution on weights, then draw `x` from `f_z`.

**Identifiability and the bimodality boundary.** Two-component
mixtures of normals are an instructive case: for homoscedastic
components (equal `sigma`), the mixture density has two modes if and
only if `|mu_1 - mu_2| > 2 * sigma` (Ray e Lindsay, 2005). Below that
threshold the joint density is unimodal even when the components are
distinct. The catalogue layer therefore does not equate "data look
unimodal" with "fit a single Tier A": even when the dip test fails to
reject unimodality, a GMM-2 fit can win on WAIC if the components
overlap heavily.

**Label switching.** Posterior densities of `(p, theta_1, ..., theta_k)`
have `k!` symmetric modes from permuting the component labels. The
catalogue mitigates by enforcing `mu_1 <= mu_2 <= ... <= mu_k` at the
post-processing stage, which is the simplest non-arbitrary
identifiability constraint. Per-chain PT is essential here: the
posterior is genuinely multimodal in the label dimension, and AM
diagonal alone cannot cross it.

The catalogue accepts any pair (or k-tuple) of Tier A
entries plus mixing weights `(p_1, ..., p_k)` summing to one. The log-
density of a `k`-component mixture is the log-sum-exp of the
component log-densities, which the DSL handles via the
`log(exp(...) + exp(...))` pattern.

Recommended primary entries:

- 2-component Gaussian mixture `GMM(p, mu1, sigma1, mu2, sigma2)`: the
  canonical bimodal case; default for any column with bimodality
  detected by the Hartigan's dip test.
- 3-component Gaussian mixture: trimodal extension.
- 2-component Lognormal mixture: bimodal positive data.
- 2-component t-Student mixture: bimodal with heavy tails.

For the 2-component GMM:

```r
gpum_loglik <- ~ log(
    p * exp(-((y - mu1)^2) / (2 * sigma1^2)) / sigma1
  + (1 - p) * exp(-((y - mu2)^2) / (2 * sigma2^2)) / sigma2
)
init        <- list(p = 0.5, mu1 = quantile(y, 0.25),
                    log_sigma1 = log(mad(y)),
                    mu2 = quantile(y, 0.75),
                    log_sigma2 = log(mad(y)))
proposal_sd <- list(p = 0.05, mu1 = sd(y)/sqrt(length(y)),
                    log_sigma1 = 0.1,
                    mu2 = sd(y)/sqrt(length(y)),
                    log_sigma2 = 0.1)
method      <- "pt"   # auto, because mixtures have multimodal posteriors
```

Label switching is mitigated by ordering the components on `mu1 < mu2`
at the posterior post-processing stage.

## Tier C: discrete

### Discrete, finite support

#### Bernoulli `Bern(p)`

```r
gpum_loglik <- ~ y * log(p) + (1 - y) * log(1 - p)
init        <- list(p = mean(y))
proposal_sd <- list(p = 0.05)
```

#### Binomial `Binom(n, p)`

`n` known from the data construction.

```r
gpum_loglik <- ~ y * log(p) + (n - y) * log(1 - p)
init        <- list(p = mean(y) / n)
proposal_sd <- list(p = 0.05)
```

#### Beta-binomial `BB(n, alpha, beta)`

Overdispersed binomial. Needs `lgamma`.

#### Hypergeometric `Hyper(N, K, n)`

All three of `N`, `K`, `n` come from sampling design, no parameter
fitting; included for the catalogue completeness rather than for
selection.

### Discrete, infinite support

#### Poisson `Poisson(lambda)`

```r
gpum_loglik <- ~ y * log(lambda) - lambda - lgamma(y + 1)
init        <- list(log_lambda = log(mean(y) + 0.5))
proposal_sd <- list(log_lambda = 0.1)
```

#### Negative binomial `NB(mu, phi)`

GLM workhorse for overdispersed counts.

#### Geometric `Geom(p)`

Number of trials until first success.

## Tier D, deferred

Multivariate Normal, multivariate t, Dirichlet, multinomial, Wishart,
Inverse Wishart, LKJ correlation. Each needs the vector and matrix
DSL of Tier 5 and is gated behind a real case pulling for it; see the
decision in `BRIEFING.md`.

## Detecting multimodality before fitting

The pipeline runs three orthogonal pre-fit checks, all on the raw
column `y`, and records the result. The detection is informative, not
gating: even a "unimodal" verdict still admits a 2-component mixture
as a challenger candidate, because heavy component overlap can hide
the second mode (see Ray-Lindsay threshold above).

### Hartigan dip test (primary)

The dip statistic `D` measures the maximum deviation between the
empirical CDF and the closest unimodal CDF, in the supremum sense.
Under the null of unimodality, `D` follows a known limit distribution
tabulated in Hartigan e Hartigan (1985). The test is implemented in
the R package `diptest` as `diptest::dip.test`, with a p-value via
Monte Carlo against a unimodal reference (uniform on the convex hull of
the data, the worst case among unimodals).

**Decision rule for the catalogue.** Reject unimodality at `p < 0.05`,
record the p-value as a column of the candidate ranking, and always
include the matched GMM-2 (or higher-`k`) as a challenger candidate.
The dip test is fast (`O(n log n)`), so it runs unconditionally on
every column.

### Silverman bootstrap (secondary)

For `H_k: data have at most k modes`, Silverman's bootstrap (Silverman,
1981) bootstraps the critical bandwidth of a kernel density estimator
and compares it to the observed bandwidth. The test is more general
than the dip test (handles arbitrary `k`) but costs an order of
magnitude more compute. Used by the catalogue only when the user asks
for `mode_count = "silverman"` or when the dip test borderlines
between `p = 0.05` and `p = 0.20`.

### Kernel density mode counting (visual)

A Gaussian KDE with bandwidth selected by Silverman's rule of thumb,
followed by sign-change counting of the first derivative. Yields an
integer mode count, no p-value, intended as the picture the user sees
on the diagnostic plot. The number is recorded as `mode_count_kde` in
the ranking table.

### Bayes-style mode count: WAIC delta

Not strictly a hypothesis test but the consistent Bayesian companion
of the dip test. For every column the catalogue runs the best Tier A
unimodal fit and the GMM-2 fit, computes `delta = WAIC(GMM-2) - WAIC(best Tier A)`,
and records the delta. A large negative delta (e.g., `< -10` on the
deviance scale, the rule of thumb of Vehtari et al. 2017) is positive
evidence for a mixture even when the dip test does not reject.

### Joint verdict

The catalogue prints, at the head of the ranking table:

```
multimodality:
  Hartigan dip test         : p = 0.0001  (reject H0: unimodal)
  KDE mode count            : 2
  WAIC(GMM-2) - WAIC(normal): -34.2       (strong evidence for mixture)
  recommended_method        : pt
```

When any of the three rejects unimodality, the catalogue activates the
mixture branch and runs the gpumetropolis fits with `method = "pt"`
automatically. Otherwise it runs with `method = "rwm"`. The user can
override.

## Ranking and selection

Each candidate from the eligible bucket is fit by the standard
gpumetropolis pipeline (`gpu_metropolis`, AM warmup or PT depending
on the family). The catalogue layer then computes:

| Criterion | Cost | Honesty | Used as |
|---|---|---|---|
| AIC | one log-likelihood at the posterior mean | freq | pre-filter, rough rank |
| BIC | same as AIC plus `+ k * log(n)` | freq, conservative | pre-filter |
| WAIC | log-pointwise on the draws | Bayes | primary rank |
| LOO-PSIS | leaving each obs out, via importance sampling | Bayes, gold | tie breaker for top three |

The primary ranking column is `WAIC`. Models are sorted ascending in
WAIC; `gpum_diagnose` opens for the top three.

## Diagnostics per fit

For every candidate, the catalogue records:

- Posterior predictive check: 200 samples `y_rep` from the fitted model,
  histogram overlay vs `y`.
- QQ plot, empirical quantiles vs fitted quantiles.
- KS statistic, empirical CDF vs fitted CDF; recorded but not used as a
  selection criterion (too restrictive for misspecified-but-useful
  models).
- Mode count match: number of modes detected in `y` (Hartigan's dip
  test plus kernel density) vs in the fitted density.
- Per-parameter convergence summary from `gpum_diagnose`.

## API sketch

```r
result <- gpum_fit_catalog(
  data    = y,
  catalog = c("normal", "lognormal", "gmm2", "gmm3", "tstudent",
              "weibull", "gamma"),
  ranking = "WAIC",
  diag    = c("ppc", "qq", "mode_count"),
  method  = "auto"
)

print(result)        # tabela: modelo, WAIC, AIC, mode_count, status
plot(result)         # top 3 com PPC + QQ
result$best          # gpum_fit do modelo melhor
result$fits$gmm2     # individual
```

`catalog = "auto"` triggers automatic bucket selection from the data:
support detection (positive/bounded/discrete), modality detection
(dip test), and proposes the eligible families. The user can override
with explicit lists or with `catalog = "all_tier_A"`.

## Open questions for the implementation

1. The `lgamma` op in the DSL: confirm CubeCL native support or write a
   minimax polynomial approximation accurate to `1e-6` over `[0.01, 1e4]`.
2. The PT activation rule for mixtures: by family (always-on for
   mixtures), by dip test (data-driven), or by `recommended_method`
   from the joint multimodality verdict. The joint verdict is the most
   defensible; pin it as the default.
3. Label switching at the catalogue layer: post-process by enforcing
   `mu_1 <= mu_2 <= ... <= mu_k` on the post-warmup draws, which is
   simple and non-arbitrary. Compare against Stephens (2000) relabelling
   if the user requests a more principled treatment.
4. The dip-test tie zone (`0.05 < p < 0.20`): default to running the
   Silverman bootstrap; allow override via
   `multimodality = c("dip", "silverman", "waic", "joint")`.
5. The Silverman bootstrap cost: parallelise over the resamples (each
   resample is independent), reuse the gpumetropolis Rayon pool.
6. WAIC computation in the kernel: stream during the sampling phase or
   post-hoc on `fit$draws`. Stream is cheaper for long runs but adds a
   kernel argument; post-hoc is simpler and fits the orchestration
   model already in place.
7. Storage of the fitted catalogue: the result of `gpum_fit_catalog`
   includes `k` fits, each a full `gpum_fit`; consider lazy
   evaluation of plots and summaries to keep memory bounded.

## References

- Forbes, C., Evans, M., Hastings, N., e Peacock, B. *Statistical
  Distributions*. 4. ed. Wiley, 2011.
- Hartigan, J. A., e Hartigan, P. M. The dip test of unimodality.
  *The Annals of Statistics*, v. 13, n. 1, p. 70-84, 1985.
- McLachlan, G., e Peel, D. *Finite Mixture Models*. Wiley, 2000.
- Ray, S., e Lindsay, B. G. The topography of multivariate normal
  mixtures. *The Annals of Statistics*, v. 33, n. 5, p. 2042-2065, 2005.
- Silverman, B. W. Using kernel density estimates to investigate
  multimodality. *Journal of the Royal Statistical Society: Series B*,
  v. 43, n. 1, p. 97-99, 1981.
- Johnson, N. L., Kotz, S., e Balakrishnan, N. *Continuous Univariate
  Distributions*. v. 1 e v. 2. Wiley, 1994 e 1995.
- Johnson, N. L., Kemp, A. W., e Kotz, S. *Univariate Discrete
  Distributions*. 3. ed. Wiley, 2005.
- Leemis, L. M., e McQueston, J. T. Univariate distribution
  relationships. *The American Statistician*, v. 62, n. 1, p. 45-53, 2008.
- Rigby, R. A., e Stasinopoulos, D. M. Generalized additive models for
  location, scale and shape. *Journal of the Royal Statistical Society:
  Series C*, v. 54, n. 3, p. 507-554, 2005.
- Vehtari, A., Gelman, A., e Gabry, J. Practical Bayesian model
  evaluation using leave-one-out cross-validation and WAIC.
  *Statistics and Computing*, v. 27, n. 5, p. 1413-1432, 2017.
- Wikipedia. *List of probability distributions*. Acessado em
  2026-06-15.
