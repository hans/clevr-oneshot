"""
MCMC inference sandbox, second model: minimal CCG model.

Data consists of two-word sequences. Generative model:
bag-of-CCG-lexical-items.

category_i ~ Dirichlet
word_j ~ category_i

Well, with a post-hoc constraint that a sentence parse.
"""

from collections import Counter
import itertools

import numpy as np
from tqdm import trange


dataset = ["the dog", "the human", "a dog", "a human", "a woman", "the dogs", "the women",
           "three women", "three dogs"]
dataset = [sentence.split() for sentence in dataset]
vocabulary = sorted(set(itertools.chain.from_iterable(dataset)))
word2idx = {word: idx for idx, word in enumerate(vocabulary)}

categories = ["N/N", "N"]


class ParserParameters(object):

  def __init__(self, parameters):
    self.p = parameters

  def __str__(self):
    return "ParserParameters<%s>" % self.p

  __repr__ = __str__

  @classmethod
  def sample(cls):
    """
    Sample random model parameters.
    """
    parameters = {}
    # Sample parameters for each category.
    for category in categories:
      parameters[category] = np.random.dirichlet((1,) * len(vocabulary))

    return ParserParameters(parameters)

  @classmethod
  def from_array(cls, array):
    parameters = {}
    for start_idx, category in zip(np.arange(0, len(categories) * len(vocabulary), len(vocabulary)), categories):
      parameters[category] = array[start_idx:start_idx + len(vocabulary)]
    return ParserParameters(parameters)

  def as_array(self):
    ret = np.empty((len(categories) * len(vocabulary),))
    for start_idx, category in zip(np.arange(0, len(categories) * len(vocabulary), len(vocabulary)), categories):
      ret[start_idx:start_idx + len(vocabulary)] = self.p[category]
    return ret


def run_mcmc(x0, proposal_fn, loglk_fn, iterations=10000):
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
  def loglk_fn(params):
    params = ParserParameters.from_array(params)

    loglk = 0
    for item in dataset:
      # Marginalize out category assignments
      assignments = itertools.product(categories, repeat=len(item))
      local_prob = 0
      for assignment in assignments:
        if assignment not in [("N/N", "N")]:
          continue

        assignment_logprob = 1
        for word, cat in zip(item, assignment):
          assignment_logprob += np.log(params.p[cat][word2idx[word]])

        local_prob += np.exp(assignment_logprob)
      loglk += np.log(local_prob)

    return loglk

  def proposal_fn(x):
    return ParserParameters.sample().as_array(), 0

  x0 = ParserParameters.sample().as_array()
  chain, loglks = run_mcmc(x0, proposal_fn, loglk_fn)

  from pprint import pprint
  params = ParserParameters.from_array(chain.mean(axis=0))

  for category in categories:
    print(category)
    pprint(list(sorted(zip(vocabulary, params.p[category]), key=lambda x: x[1], reverse=True)))
