// Target model M3 for the Stan backend: a heavy-tailed location model.
// y ~ Student-t(nu, mu, scale), improper uniform prior on mu.
data {
  int<lower=1> N;
  vector[N] y;
  real<lower=0> nu;
  real<lower=0> scale;
}
parameters {
  real mu;
}
model {
  y ~ student_t(nu, mu, scale);
}
