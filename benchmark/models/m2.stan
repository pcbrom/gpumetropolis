// Target model M2 for the Stan backend: a separated bimodal posterior.
// y ~ Normal(|mu|, sigma), sigma known, improper uniform prior on mu.
data {
  int<lower=1> N;
  vector[N] y;
  real<lower=0> sigma;
}
parameters {
  real mu;
}
model {
  y ~ normal(abs(mu), sigma);
}
