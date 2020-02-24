data {
  int<lower=1> N;
  vector[N] repN;
  vector[N] RT;
}

parameters {
  real a;
  real<lower=0> b;
  real<lower=0> c;
  real<lower=0> sigma;
}

model {
  vector[N] mu;
  for (n in 1:N) {
    mu[n] = a + b * repN[n]^(-c);
  }

  target += normal_lpdf(RT | mu, sigma);
  target += normal_lpdf(a | 0, 10);
  target += normal_lpdf(b | 0, 10);
  target += normal_lpdf(c | 0, 10);
  target += cauchy_lpdf(sigma | 0, 10) - cauchy_lccdf(0 | 0, 10);
}
