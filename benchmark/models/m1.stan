// Target model M1 for the Stan backend: Gaussian mean with known sigma and a
// flat prior on mu. With no prior statement, Stan places an improper uniform
// prior on mu, which matches the flat prior used by the other backends.
data {
  int<lower=1> N;
  vector[N] y;
  real<lower=0> sigma;
}
parameters {
  real mu;
}
model {
  y ~ normal(mu, sigma);
}
