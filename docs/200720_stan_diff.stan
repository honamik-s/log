data {
  int<lower=1> N;      // 参加者数
  int C;               // 条件数
  
  int<lower=0> N_Re_Te_max; // "E"反応数の最大値, 標的=E
  int<lower=0> N_Rf_Te_max; // "F"反応数の最大値, 標的=E
  int<lower=0> N_Re_Tf_max; // "E"反応数の最大値, 標的=F
  int<lower=0> N_Rf_Tf_max; // "F"反応数の最大値, 標的=F
  
  int<lower=0> N_Re_Te[N, C];  // "E"反応数, 標的=E
  int<lower=0> N_Rf_Te[N, C];  // "F"反応数, 標的=E
  int<lower=0> N_Re_Tf[N, C];  // "E"反応数, 標的=F
  int<lower=0> N_Rf_Tf[N, C];  // "F"反応数, 標的=F

  real RT_Re_Te[N, C, N_Re_Te_max];  // "E"反応をした試行の反応時間, 標的=E
  real RT_Rf_Te[N, C, N_Rf_Te_max];  // "F"反応をした試行の反応時間, 標的=E
  real RT_Re_Tf[N, C, N_Re_Tf_max];  // "E"反応をした試行の反応時間, 標的=F
  real RT_Rf_Tf[N, C, N_Rf_Tf_max];  // "F"反応をした試行の反応時間, 標的=F
  
  real minRT[N];       // 各参加者の反応時間の最小値
  real RTbound;        // 反応時間の打ち切り値
}

parameters {
  vector[3] mu_p;
  vector<lower=0>[3] sigma;
  
  vector<lower=0>[N] alpha_pr[2];
  vector<lower=0>[N] delta_pr[2];
  vector<lower=RTbound,upper=max(minRT)>[N] tau_pr[2];
}

transformed parameters {
  vector<lower=0>[N] alpha[2]; // 閾値
  vector<lower=0>[N] delta[2]; // ドリフト率
  vector<lower=RTbound, upper=max(minRT)>[N] tau[2]; // 非決定時間
  
  for (c in 1:C){
    alpha[c] = exp(mu_p[1] + sigma[1] * alpha_pr[c]);
    delta[c] = exp(mu_p[2] + sigma[2] * delta_pr[c]);
    for (n in 1:N) {
      tau[c][n]  = Phi_approx(mu_p[3] + sigma[3] * tau_pr[c][n]) * (minRT[n]-RTbound) + RTbound;
    }
  }
}

model {
  mu_p  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  for(c in 1:C){
   alpha_pr[c] ~ normal(0, 1);
   delta_pr[c] ~ normal(0, 1);
   tau_pr[c]   ~ normal(0, 1);
  }

  for (n in 1:N) {
    for (c in 1:C) {
      // "E"反応, 標的=E
      target += wiener_lpdf(RT_Re_Te[n,c, :N_Re_Te[n,c]] | alpha[c][n], tau[c][n], 0.5, delta[c][n]);
      if(N_Rf_Te[n, c]!=0){
      // "F"反応, 標的=E
        target += wiener_lpdf(RT_Rf_Te[n,c, :N_Rf_Te[n,c]] | alpha[c][n], tau[c][n], 0.5, -delta[c][n]);
        }
      // "F"反応, 標的=F
      target += wiener_lpdf(RT_Rf_Tf[n,c, :N_Rf_Tf[n,c]] | alpha[c][n], tau[c][n], 0.5, delta[c][n]);
      if(N_Re_Tf[n,c]!=0){
      // "E"反応, 標的=F
        target += wiener_lpdf(RT_Re_Tf[n,c, :N_Re_Tf[n,c]] | alpha[c][n], tau[c][n], 0.5, -delta[c][n]);
      }
    }
  }
}