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
logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)

from nltk.ccg import lexicon as ccg_lexicon
from nltk.sem.logic import Expression
import numpy as np
from tqdm import trange

from clevros.lexicon import Lexicon, Token
from clevros.chart import WeightedCCGChartParser, printCCGDerivation
from clevros.logic import TypeSystem, Ontology
from clevros.model import Model


types = TypeSystem(["obj", "boolean"])
functions = [
  types.new_function("book", ("obj", "boolean"), lambda x: x.type == "book"),
  types.new_function("rock", ("obj", "boolean"), lambda x: x.type == "rock"),
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
  ("rock", scene, objs[1]),
  ("book", scene, objs[0]),
]

#####

vocabulary = sorted(set(itertools.chain.from_iterable(
                words.split() for words, _, _ in examples)))

lex = Lexicon.fromstring(r"""
  :- N
""", include_semantics=True)
categories = [lex._start]

sem_tokens = [f.name for f in functions]


class ParserParameters(object):
  """
  Stores a token parameter setting for the parser.

  Members:
    p_cat: `n_category`-ndarray, prior distribution over syntactic
      categories
    p_tokens_cat: `n_category * n_vocabulary`-ndarray, conditional
      distributions P(token | category)
    p_sems_token: `n_vocabulary * n_sem`-ndarray, conditional distributions
      P(sem_token | token)
  """

  def __init__(self, categories, vocabulary, sem_tokens,
               p_cat, p_tokens_cat, p_sems_token):
    self.categories = categories
    self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
    self.vocabulary = vocabulary
    self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    self.sem_tokens = sem_tokens
    self.sem2idx = {sem: idx for idx, sem in enumerate(self.sem_tokens)}

    self.p_cat = p_cat
    self.p_tokens_cat = p_tokens_cat
    self.p_sems_token = p_sems_token

  @classmethod
  def sample(cls, categories, vocabulary, sem_tokens):
    p_cat = np.random.dirichlet((1,) * len(categories))
    p_tokens_cat = np.random.dirichlet((1,) * len(vocabulary),
                                       size=len(categories))
    p_sems_token = np.random.dirichlet((1,) * len(sem_tokens),
                                       size=len(vocabulary))
    return cls(categories, vocabulary, sem_tokens,
               p_cat, p_tokens_cat, p_sems_token)

  @classmethod
  def average(cls, instances):
    """
    Create a parameter instance which is the average of many sampled instances.
    """
    p_cat = np.mean([inst.p_cat for inst in instances], axis=0)
    p_tokens_cat = np.mean([inst.p_tokens_cat for inst in instances], axis=0)
    p_sems_token = np.mean([inst.p_sems_token for inst in instances], axis=0)

    inst0 = instances[0]
    return cls(inst0.categories, inst0.vocabulary, inst0.sem_tokens,
               p_cat, p_tokens_cat, p_sems_token)

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
      logp += np.log(self.p_sems_token[word_id, sem_id])
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
    lex = Lexicon(str(self.categories[0]), [str(cat) for cat in self.categories],
                  {}, {})
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


def run_mcmc(x0, proposal_fn, loglk_fn, iterations=5000):
  """
  Args:
    x0: initial parameter setting
    proposal_fn: Function mapping from x -> (x*, log_density_ratio)
    loglk_fn: Function computing log-likelihood of x assignment
    iterations: # iterations to run.
  """

  chain = [None] * (iterations + 1)
  loglks = np.empty((iterations + 1,))

  x = x0
  chain[0] = x0
  loglks[0] = loglk_fn(x0)
  n_accepts = 0

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
      n_accepts += 1
    else:
      loglks[i] = loglks[i - 1]

    chain[i] = x

  L.info("Number of proposals accepted: %d/%d (%f%%)" % (n_accepts, iterations, n_accepts / iterations * 100))
  return chain, loglks


if __name__ == '__main__':
  def loglk_fn(weights):
    lex = weights.as_lexicon(ontology)
    parser = WeightedCCGChartParser(lex)

    loglk = 0
    for utt, scene, answer in examples:
      weighted_results = parser.parse(utt.split(), return_aux=True)
      model = Model(scene, ontology)

      local_prob = 0
      for parse, logp, _ in weighted_results:
        root_token, _ = parse.label()
        if model.evaluate(root_token.semantics()) == answer:
          local_prob += np.exp(logp)

      loglk += np.log(local_prob)

    return loglk

  def proposal_fn(x):
    sample = ParserParameters.sample(categories, vocabulary, sem_tokens)
    log_density_ratio = 0.0 # TODO
    return sample, log_density_ratio

  x0, _ = proposal_fn(None)
  chain, loglks = run_mcmc(x0, proposal_fn, loglk_fn)

  from pprint import pprint
  # Try something
  params = ParserParameters.average(chain)
  print(params.p_tokens_cat)

  pprint({word: sorted([(entry, entry.weight()) for entry in entries],
                       key=lambda x: x[1], reverse=True)
          for word, entries in params.as_lexicon(ontology)._entries.items()})
