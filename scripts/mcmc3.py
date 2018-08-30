"""
MCMC inference sandbox, second model: minimal CCG model,
with an option to split words into two primitive categories.
The setup: examples are of the form "a(n) X"; an optimal
model would learn to separate Xs beginning with vowels and
Xs beginning with consonants into two separate CCG categories.

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

from clevros.lexicon import Lexicon, Token
from clevros.chart import WeightedCCGChartParser, printCCGDerivation


dataset = ["a dog", "a bus", "a show", "a cat",
           "an apple", "an idea", "an image", "an extra"]
dataset = [sentence.split() for sentence in dataset]
vocabulary = sorted(set(itertools.chain.from_iterable(dataset)))
word2idx = {word: idx for idx, word in enumerate(vocabulary)}

lex = Lexicon("N", ["N", "NA", "NB"], [], [])
categories = [lex.parse_category("NA"), lex.parse_category("NB"),
              lex.parse_category("N/NA"), lex.parse_category("N/NB")]


def weights_from_lexicon(lex):
  weights = np.empty((len(categories), len(vocabulary)))
  cat2idx = {cat: idx for idx, cat in enumerate(categories)}
  for j, word in enumerate(vocabulary):
    entries_j = lex._entries[word]
    for entry in entries_j:
      i = cat2idx[entry.categ()]
      weights[i, j] = entry.weight()
  return weights

def lexicon_from_weights(weights):
  # NB no copy
  entries = {word: [Token(word, category, weight=weight)
                    for category, weight
                    in zip(categories, weights[:, word2idx[word]])]
             for word in vocabulary}

  lex._entries = entries
  return lex


def run_mcmc(x0, proposal_fn, loglk_fn, iterations=5000):
  """
  Args:
    x0: initial parameter setting
    proposal_fn: Function mapping from x -> (x*, log_density_ratio)
    loglk_fn: Function computing log-likelihood of x assignment
    iterations: # iterations to run.
  """

  chain = np.empty((iterations + 1,) + x0.shape)
  loglks = np.empty((iterations + 1,))

  x = x0
  chain[0] = x0
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

    chain[i] = x

  return chain, loglks


if __name__ == '__main__':
  def loglk_fn(weights):
    lex = lexicon_from_weights(weights)
    parser = WeightedCCGChartParser(lex)

    loglk = 0
    for item in dataset:
      weighted_results = parser.parse(item, return_aux=True)
      local_prob = 0
      for parse, logp, _ in weighted_results:
        local_prob += np.exp(logp)

      loglk += np.log(local_prob)

    return loglk

  def proposal_fn(x):
    # Symmetric proposal
    return (np.random.dirichlet((1,) * len(vocabulary),
                                size=len(categories)),
            0.0)

  x0, _ = proposal_fn(None)
  chain, loglks = run_mcmc(x0, proposal_fn, loglk_fn)

  from pprint import pprint
  params = lexicon_from_weights(chain.mean(axis=0))
  pprint({word: sorted([(entry, entry.weight()) for entry in entries],
                       key=lambda x: x[1], reverse=True)
          for word, entries in params._entries.items()})
