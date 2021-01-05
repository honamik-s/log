data {
  int<lower=1> N;
  vector[N] setsize;
  vector[N] RT;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  vector[N] mu = alpha + beta * setsize;
  target += normal_lpdf(RT | mu, sigma);
  target += normal_lpdf(alpha | 0, 10);
  target += normal_lpdf(beta | 0, 10);
  target += cauchy_lpdf(sigma | 0, 10) - cauchy_lccdf(0 | 0, 10);
}
