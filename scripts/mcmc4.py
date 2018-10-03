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
from frozendict import frozendict
import itertools
import logging
import random
logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)

from nltk.ccg import lexicon as ccg_lexicon
from nltk.ccg.api import PrimitiveCategory
from nltk.sem.logic import Expression
import numpy as np
from scipy import stats
from tqdm import trange

from clevros.lexicon import Lexicon, Token
from clevros.chart import WeightedCCGChartParser, printCCGDerivation, ApplicationRuleSet
from clevros.logic import TypeSystem, Ontology
from clevros.model import Model
from clevros.primitives import fn_unique


types = TypeSystem(["obj", "boolean"])
functions = [
  types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),
  types.new_function("book", ("obj", "boolean"), lambda x: x["type"] == "book"),
  types.new_function("rock", ("obj", "boolean"), lambda x: x["type"] == "rock"),
]
ontology = Ontology(types, functions, constants=[])

#####

scene = {
  "objects": [
    frozendict({"type": "book"}),
    frozendict({"type": "rock"}),
  ]
}
objs = scene["objects"]

examples = [
    ("the rock", scene, objs[1]),
    ("the stone", scene, objs[1]),
    ("the book", scene, objs[0]),
    ("the tome", scene, objs[0]),
]

#####

vocabulary = sorted(set(itertools.chain.from_iterable(
                words.split() for words, _, _ in examples)))

lex = Lexicon.fromstring(r"""
  :- N, NA, NB
""", include_semantics=True)
categories = [lex.parse_category(x) for x in ["N", "NA", "NB", "N/NA", "N/NB"]]

sem_tokens = [f.name for f in functions]


class ParserParameters(object):
  """
  Stores a token parameter setting for the parser.

  Members:
    p_cat: `n_category`-ndarray, prior distribution over syntactic
      categories
    p_tokens_cat: `n_category * n_vocabulary`-ndarray, conditional
      distributions P(token | category)
    p_sems_cat: `n_category * n_sem`-ndarray, conditional distributions
      P(sem_token | category)
  """

  def __init__(self, categories, vocabulary, sem_tokens,
               p_cat, p_tokens_cat, p_sems_cat):
    self.categories = categories
    self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
    self.vocabulary = vocabulary
    self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    self.sem_tokens = sem_tokens
    self.sem2idx = {sem: idx for idx, sem in enumerate(self.sem_tokens)}

    self.p_cat = p_cat
    self.p_tokens_cat = p_tokens_cat
    self.p_sems_cat = p_sems_cat

  @classmethod
  def sample(cls, categories, vocabulary, sem_tokens):
    density = 0.0

    cat_prior = (100,) * len(categories)
    p_cat = np.random.dirichlet(cat_prior)
    density += stats.dirichlet.logpdf(p_cat, cat_prior)

    tokens_cat_prior = (100,) * len(vocabulary)
    p_tokens_cat = np.random.dirichlet(tokens_cat_prior,
                                       size=len(categories))
    for sample in p_tokens_cat:
      density += stats.dirichlet.logpdf(sample, tokens_cat_prior)

    sems_cat_prior = (100,) * len(sem_tokens)
    p_sems_cat = np.random.dirichlet(sems_cat_prior,
                                     size=len(categories))
    for sample in p_sems_cat:
      density += stats.dirichlet.logpdf(sample, sems_cat_prior)

    return (cls(categories, vocabulary, sem_tokens,
                p_cat, p_tokens_cat, p_sems_cat),
            density)

  @classmethod
  def average(cls, instances):
    """
    Create a parameter instance which is the average of many sampled instances.
    """
    p_cat = np.mean([inst.p_cat for inst in instances], axis=0)
    p_tokens_cat = np.mean([inst.p_tokens_cat for inst in instances], axis=0)
    p_sems_cat = np.mean([inst.p_sems_cat for inst in instances], axis=0)

    inst0 = instances[0]
    return cls(inst0.categories, inst0.vocabulary, inst0.sem_tokens,
               p_cat, p_tokens_cat, p_sems_cat)

  def noise(self, scaling_factor=100):
    # Resample parameters from Dirichlet priors parameterized by the current
    # weights.
    D = stats.dirichlet
    density = 0.0

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

    # Noise just one of the parameters.
    which_parameter = np.random.choice(["cat", "tokens_cat"], p=[0.25, 0.75])

    if which_parameter == "cat":
      p_cat = D_sample(self.p_cat * scaling_factor)
    else:
      p_cat = self.p_cat
    density += D.logpdf(p_cat, self.p_cat * scaling_factor)

    if which_parameter == "tokens_cat":
      # Resample just an individual category -> word distribution.
      cat_id = np.random.choice(len(self.categories))

      p_tokens_cat = self.p_tokens_cat.copy()
      p_tokens_cat[cat_id] = D_sample(self.p_tokens_cat[cat_id] * scaling_factor)
    else:
      p_tokens_cat = self.p_tokens_cat
    print(which_parameter)
    for sample, dir_prior in zip(p_tokens_cat, self.p_tokens_cat):
      density += D.logpdf(sample, dir_prior * scaling_factor)

    # HACK: don't resample p_sems_cat
    # p_sems_cat = np.array([D_sample(dir_prior * scaling_factor)
    #                         for dir_prior in self.p_sems_cat])
    # for sample, dir_prior in zip(p_sems_cat, self.p_sems_cat):
    #   density += D.logpdf(sample, dir_prior * scaling_factor)
    p_sems_cat = self.p_sems_cat.copy()

    return (ParserParameters(self.categories, self.vocabulary, self.sem_tokens,
                             p_cat, p_tokens_cat, p_sems_cat),
            density)

  def score_token(self, token):
    cat_id = self.cat2idx[token.categ()]
    word_id = self.word2idx[token._token]

    # Category prior
    logp = np.log(self.p_cat[cat_id])
    # token | category
    logp += np.log(self.p_tokens_cat[cat_id, word_id])
    # sem | token
    for predicate in token.semantics().predicates():
      sem_id = self.sem2idx[predicate.name]
      logp += np.log(self.p_sems_cat[cat_id, sem_id])
    return logp

  def score_parse(self, parse):
    score = sum(self.score_token(token) for _, token in parse.pos())
    return score

  def as_lexicon(self, ontology, max_sem_depth=3):
    """
    Convert parameters to a `Lexicon` instance in order to pass to parser.
    It's also an expensive exhaustive conversion which won't scale!

    Args:
      ontology: Ontology instance which iterates over legal logical expressions
    """
    ccg_lexicon.CCGVar.reset_id()
    primitives = [str(cat) for cat in self.categories
                  if isinstance(cat, PrimitiveCategory)]
    lex = Lexicon(str(self.categories[0]), primitives, {}, {})
    for word in self.vocabulary:
      w_entries = []
      for category in self.categories:
        for expr in ontology.iter_expressions(max_depth=max_sem_depth):
          token = Token(word, category, semantics=expr)
          weight = self.score_token(token)
          token._weight = weight
          w_entries.append(token)

      lex._entries[word] = w_entries

    return lex


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

        import pandas as pd
        df = pd.DataFrame(x.p_tokens_cat, columns=vocabulary, index=list(map(str, categories)))
        print(df)
      else:
        loglks[i] = loglks[i - 1]
        proposal_densities[i] = proposal_densities[i - 1]

      chain[i] = x

      t.set_postfix(accept="%2.2f" % (n_accepts / (i + 1) * 100))

  L.info("Number of proposals accepted: %d/%d (%f%%)" % (n_accepts, iterations, n_accepts / iterations * 100))
  return chain, loglks


if __name__ == '__main__':
  def loglk_fn(weights):
    lex = weights.as_lexicon(ontology)
    parser = WeightedCCGChartParser(lex, ruleset=ApplicationRuleSet)

    loglk = 0
    for utt, scene, answer in examples:
      weighted_results = parser.parse(utt.split(), scorer=weights.score_parse,
                                      return_aux=True)
      model = Model(scene, ontology)

      local_prob = 0
      for parse, logp, _ in weighted_results:
        root_token, _ = parse.label()
        if model.evaluate(root_token.semantics()) == answer:
          local_prob += np.exp(logp)

      loglk += np.log(local_prob)

    return loglk

  def proposal_fn(weights):
    if weights is None:
      sample, density = ParserParameters.sample(categories, vocabulary, sem_tokens)
    else:
      sample, density = weights.noise()
    return sample, density

  x0, _ = proposal_fn(None)

  # HACK: fix some known parameters
  def one_hot(i, n):
    ret = np.zeros(n) + 5e-2
    ret[i] = 1.0
    ret /= ret.sum()
    return ret
  n_sems = len(x0.sem2idx)
  x0.p_sems_cat[x0.cat2idx[PrimitiveCategory("NA")]] = one_hot(x0.sem2idx["book"], n_sems)
  x0.p_sems_cat[x0.cat2idx[PrimitiveCategory("NB")]] = one_hot(x0.sem2idx["rock"], n_sems)
  x0.p_sems_cat[x0.cat2idx[lex.parse_category("N/NA")]] = one_hot(x0.sem2idx["unique"], n_sems)
  x0.p_sems_cat[x0.cat2idx[lex.parse_category("N/NB")]] = one_hot(x0.sem2idx["unique"], n_sems)

  noun_dist = one_hot(x0.cat2idx[PrimitiveCategory("NA")], len(categories)) + \
      one_hot(x0.cat2idx[PrimitiveCategory("NB")], len(categories))
  noun_dist /= noun_dist.sum()
  x0.p_tokens_cat[:, x0.word2idx["book"]] = noun_dist
  x0.p_tokens_cat[:, x0.word2idx["tome"]] = noun_dist
  x0.p_tokens_cat[:, x0.word2idx["rock"]] = noun_dist
  x0.p_tokens_cat[:, x0.word2idx["stone"]] = noun_dist
  x0.p_tokens_cat[:, x0.word2idx["the"]] = 0.1
  x0.p_tokens_cat[x0.cat2idx[lex.parse_category("N/NA")], x0.word2idx["the"]] = 0.4
  x0.p_tokens_cat[x0.cat2idx[lex.parse_category("N/NB")], x0.word2idx["the"]] = 0.4
  x0.p_tokens_cat /= x0.p_tokens_cat.sum(axis=1, keepdims=True)
  print(x0.p_tokens_cat)

  chain, loglks = run_mcmc(x0, proposal_fn, loglk_fn)

  from pprint import pprint
  # Try something
  params = ParserParameters.average(chain)
  print(params.p_tokens_cat)

  pprint({word: sorted([(entry, entry.weight()) for entry in entries],
                       key=lambda x: x[1], reverse=True)
          for word, entries in params.as_lexicon(ontology)._entries.items()})
