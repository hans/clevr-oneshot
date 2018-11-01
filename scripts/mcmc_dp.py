"""
Basic MH sampling for a DP mixture model.
"""

from collections import Counter
from frozendict import frozendict
import itertools
import logging
import random
logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)

import numpy as np
import scipy as sp
from scipy import stats
from tqdm import trange


# Gamma sampling method borks when alpha parameters are very small. Below
# we define a Dirichlet sampler based on Beta samples, which is slower but
# more stable.
def D_sample(alphas, size=None):
  try:
    ret = D.rvs(alphas, size=size)
  except ValueError:
    ret = [np.random.beta(alphas[0], sum(alphas[1:]))]
    for j in range(1, len(alphas) - 1):
      phi = np.random.beta(alphas[j], sum(alphas[j + 1:]))
      ret.append((1 - sum(ret)) * phi)
    ret.append(1 - sum(ret))
    ret = np.array(ret)

  if ret.min() == 0:
    ret += 1e-15
    ret /= ret.sum()
  return ret


def run_mcmc(x0, proposal_fn, loglk_fn, iterations=300):
  """
  Args:
    x0: initial parameter setting
    proposal_fn: Function mapping from x -> (x*, log_density_ratio)
    loglk_fn: Function computing log-likelihood of x assignment
    iterations: # iterations to run.
  """

  chain = [None] * (iterations + 1)
  proposal_densities = np.empty((iterations + 1,))
  loglks = np.empty((iterations + 1,))

  x = x0
  chain[0] = x0
  loglks[0] = loglk_fn(x0)
  proposal_densities[0] = np.log(1e-10)
  n_accepts = 0

  with trange(1, iterations + 1) as t:
    for i in t:
      # draw proposal
      x_next, log_density = proposal_fn(x)
      log_proposal_ratio = log_density - proposal_densities[i - 1]

      # compute acceptance factor
      loglk_next = loglk_fn(x_next)
      loglk_ratio = loglk_next - loglks[i - 1]
      acceptance_factor = min(1, np.exp(loglk_ratio + log_proposal_ratio))
      print(log_density, proposal_densities[i - 1], log_density - proposal_densities[i - 1], loglk_next, acceptance_factor)

      if np.random.uniform() < acceptance_factor:
        # Accept proposal.
        x = x_next
        loglks[i] = loglk_next
        proposal_densities[i] = log_density
        n_accepts += 1
      else:
        loglks[i] = loglks[i - 1]
        proposal_densities[i] = proposal_densities[i - 1]

      chain[i] = x

      t.set_postfix(accept="%2.2f" % (n_accepts / (i + 1) * 100))

  L.info("Number of proposals accepted: %d/%d (%f%%)" % (n_accepts, iterations, n_accepts / iterations * 100))
  return chain, loglks


if __name__ == '__main__':
  import pandas as pd
  old_faithful_df = pd.read_csv('old_faithful.csv')
  xs = old_faithful_df.std_waiting

  ALPHA = 0.5
  K = 2

  def stick_breaking(beta):
    """
    Compute stick-breaking process distribution for given stick breaking
    weights.
    """
    portion_remaining = np.concatenate([[1], np.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

  def loglk_fn(weights):
    betas, lambdas, mus = weights
    loglk = 0

    # stick-breaking weights
    for beta in betas:
      loglk += stats.beta.logpdf(beta, 1, ALPHA)
    mixture_weights = stick_breaking(beta)

    for lambda_, mu in zip(lambdas, mus):
      # prior over lambda
      loglk += np.log(1)
      # prior over mu
      loglk += stats.norm.logpdf(mu, 0, lambda_)

    # mixture likelihood
    for x in xs:
      for weight, lambda_, mu in zip(mixture_weights, lambdas, mus):
        loglk += np.log(weight) + stats.norm.logpdf(x, mu, lambda_)

    return loglk

  def sigmoid(x):
    return 1. / (1 + np.exp(-x))

  def beta_logp(a, b, x):
    logp = 0
    if a != 1:
      logp += (a - 1) * np.log(x)
    if b != 1:
      logp += (b - 1) * np.log(1 - x)

    gammaln = sp.special.gammaln
    Z = gammaln(a) + gammaln(b) - gammaln(a + b)
    logp -= Z

    return logp

  def proposal_fn(weights):
    lambda_min, lambda_max = 0, 10

    if weights is None:
      betas = stats.beta.rvs(1, ALPHA, size=K)
      lambdas = stats.uniform.rvs(0, 1, size=K)
      mus = stats.norm.rvs(0, 3, size=K)

      weights = (betas, lambdas, mus)
      return weights, np.log(1.0)
    else:
      betas, lambdas, mus = weights

    # inv logit transform on unit continuous variables
    beta_logodds = np.log(betas / (1 - betas))
    # inv interval transform
    lambdas_ = np.log(lambdas - lambda_min) - np.log(lambda_max - lambdas)

    # Sample drifts.
    d_beta_logodds = stats.norm.rvs(0, 5, size=K)
    d_lambda = stats.norm.rvs(0, 1, size=K)
    d_mu = stats.norm.rvs(0, 1, size=K)

    # Compute drifted + transformed RVs.
    betas = sigmoid(beta_logodds + d_beta_logodds)
    lambdas = sigmoid(lambdas_ + d_lambda) * (lambda_max - lambda_min) + lambda_min
    mus = mus + d_mu
    sample = (betas, lambdas, mus)

    density = 0
    for beta in betas:
      density += beta_logp(beta, 1, ALPHA)#stats.beta.logpdf(beta, 1, ALPHA)
    for lambda_ in lambdas:
      density += np.log(1.0)
    for mu in mus:
      density += stats.norm.logpdf(mu, 0, 1)

    return sample, density

  ################

  x0, _ = proposal_fn(None)

  chain, loglks = run_mcmc(x0, proposal_fn, loglk_fn)
