data {
  int<lower=1> N_obs;              // Number of observations
  int<lower=1> N_persons;          // Number of persons
  int<lower=1> N_items;            // Number of items
  array[N_obs] int<lower=1> jj;    // Person indices
  array[N_obs] int<lower=1> kk;    // Item indices
  array[N_obs] int<lower=0, upper=1> y;  // Responses
}

parameters {
  // Person parameters
  vector[N_persons] theta;
  real mu_theta;                   // Mean ability of population
  real<lower=0> sigma_theta;       // SD of population

  // Item parameters
  vector[N_items - 1] delta_raw;    // K-1 free parameters for sum-to-zero
}

transformed parameters {
  vector[N_items] delta;
  // Apply sum-to-zero constraint on items
  // This matches Winsteps/CMLE convention
  delta[1:(N_items - 1)] = delta_raw;
  delta[N_items] = -sum(delta_raw);
}

model {
  // Hyperpriors
  mu_theta ~ normal(0, 5);
  sigma_theta ~ cauchy(0, 2);
  
  // Hierarchical prior on persons
  theta ~ normal(mu_theta, sigma_theta);
  
  // Weak prior on items
  delta_raw ~ normal(0, 3); 

  // Likelihood
  // Optimized vectorized operation
  y ~ bernoulli_logit(theta[jj] - delta[kk]);
}

generated quantities {
  // Calculate log_lik for LOO/WAIC
  vector[N_obs] log_lik;
  // Generate replicated data for PPC
  array[N_obs] int<lower=0, upper=1> y_rep;
  
  for (n in 1:N_obs) {
    real logit_p = theta[jj[n]] - delta[kk[n]];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | logit_p);
    y_rep[n] = bernoulli_logit_rng(logit_p);
  }
}
