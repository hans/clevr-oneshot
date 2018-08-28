"""
MCMC inference sandbox, first model: simple binary-choice word model.
"""

from collections import Counter

import numpy as np
from tqdm import trange

vocabulary = ["a", "b"]
dataset = (["a"] * 50) + (["b"] * 25)


def run_mcmc(x0, proposal_fn, loglk_fn, iterations=100):
  """
  Args:
    x0: initial parameter setting
    proposal_fn: Function mapping from x -> (x*, log_density_ratio)
    loglk_fn: Function computing log-likelihood of x assignment
    iterations: # iterations to run.
  """

  ndim = x0.shape[-1]
  chain = np.empty((iterations + 1, ndim))
  loglks = np.empty((iterations + 1,))

  x = x0
  chain[0, :] = x0
  loglks[0] = loglk_fn(x0)

  for i in trange(1, iterations + 1):
    # draw proposal
    x_next, log_proposal_ratio = proposal_fn(x)

    # compute acceptance factor
    loglk_next = loglk_fn(x_next)
    loglk_ratio = loglk_next - loglks[i - 1]
    acceptance_factor = np.exp(loglk_ratio + log_proposal_ratio)

    if np.random.uniform() < acceptance_factor:
      # Accept proposal.
      x = x_next
      loglks[i] = loglk_next
    else:
      loglks[i] = loglks[i - 1]

    chain[i, :] = x

  return chain, loglks


if __name__ == '__main__':
  def loglk_fn(param):
    dataset_freqs = Counter(dataset)
    log_param = np.log(param)
    log_opp = np.log(1 - param)

    return dataset_freqs["a"] * log_param + dataset_freqs["b"] * log_opp

  def proposal_fn(x):
    sample = np.random.randn(x.shape[-1])
    # Symmetric proposal; ratio is 1 ==> log-ratio is 0
    log_ratio = 0
    return sample, log_ratio

  x0 = np.array([0.5])
  chain, loglks = run_mcmc(x0, proposal_fn, loglk_fn)

  print(chain.mean(), chain.std())
