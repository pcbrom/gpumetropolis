// Target model M4 for the Stan backend: an ill-conditioned multivariate
// Gaussian target N(0, Sigma), supplied through its precision matrix P.
data {
  matrix[3, 3] P;
}
parameters {
  vector[3] theta;
}
model {
  theta ~ multi_normal_prec(rep_vector(0, 3), P);
}
