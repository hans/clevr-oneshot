"""
Reproduce causative/non-causative syntactic bootstrapping from Naigles (1990).
"""


from argparse import ArgumentParser
from copy import copy
import logging
import math
from pathlib import Path
import sys

from frozendict import frozendict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clevros.compression import Compressor
from clevros.lexicon import Lexicon, get_candidate_categories, predict_zero_shot, get_yield, set_yield
from clevros.logic import TypeSystem, Ontology
from clevros.model import Model
from clevros.primitives import *
from clevros.util import Distribution
from clevros.word_learner import WordLearner

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)

# EC hyperparameters.
EC_kwargs = {
  "topK": 1,
  "pseudoCounts": 1.0,
  "a": 1,
  "aic": -1.0,
  "structurePenalty": 0.001,
  "backend": "pypy",
  "CPUs": 1,
}

######
# Ontology.

class Cause(Action):
  def __init__(self, agent, behavior):
    self.agent = agent
    self.behavior = behavior

  def __hash__(self):
    return hash((self.agent, self.behavior))

  def __str__(self):
    return "Cause(%s, %s)" % (self.agent, self.behavior)

  __repr__ = __str__


class Move(Action):
  def __init__(self, agent, manner):
    self.agent = agent
    self.manner = manner

  def __hash__(self):
    return hash((self.agent, self.manner))

  def __str__(self):
    return "Move(%s, %s)" % (self.agent, self.manner)

  __repr__ = __str__


class Become(Action):
  def __init__(self, agent, state):
    self.agent = agent
    self.state = state

  def __hash__(self):
    return hash((self.agent, self.state))

  def __str__(self):
    return "Become(%s -> %s)" % (self.agent, self.state)

  __repr__ = __str__


types = TypeSystem(["agent", "action", "manner", "state", "boolean"])

functions = [
  types.new_function("cause", ("agent", "action", "action"), Cause),
  types.new_function("move", ("agent", "manner", "action"), Move),
  types.new_function("become", ("agent", "state", "action"), Become),

  types.new_function("unique", (("agent", "boolean"), "agent"), fn_unique),

  types.new_function("rabbit", ("agent", "boolean"), lambda a: a["type"] == "rabbit"),
  types.new_function("duck", ("agent", "boolean"), lambda a: a["type"] == "duck"),
]

constants = [
  types.new_constant("bend", "manner"),
  types.new_constant("lift", "manner"),
  types.new_constant("tilt", "manner"),

  types.new_constant("clean", "state"),
  types.new_constant("dirty", "state"),
]

ontology = Ontology(types, functions, constants)

######
# Lexicon.

initial_lexicon = Lexicon.fromstring(r"""
  :- S, N

  the => N/N {\x.unique(x)}
  duck => N {\x.duck(x)}
  bunny => N {\x.rabbit(x)}

  bends => S\N/N {\p a.cause(a,move(p,bend))}
  lifts => S\N/N {\p a.cause(a,move(p,lift))}
  tilts => S\N/N {\p a.cause(a,move(p,tilt))}
  cleans => S\N/N {\p a.cause(a,become(p,clean))}
  dirties => S\N/N {\p a.cause(a,become(p,dirty))}

  ducks => S\N {\a.move(a,bend)}
  jumps => S\N {\a.move(a,lift)}
  spins => S\N {\a.move(a,tilt)}
""", ontology, include_semantics=True)

######
# Evaluation data.
L.info("Preparing evaluation data.")

Rabbit = frozendict(type="rabbit")
Duck = frozendict(type="duck")


class Scene(object):
  def __init__(self, objects):
    self.objects = objects

  # backwards-compat
  def __getitem__(self, key):
    if key == "objects":
      return self.objects
    raise TypeError("'Scene' object is not subscriptable")

  def update_obj(self, obj, new_obj):
    idx = self.objects.index(obj)

    objects = copy(self.objects)
    objects[idx] = new_obj
    return Scene(objects)

scene = Scene([ Rabbit, Duck ])
objs = scene.objects

event = Event()

examples = [

  ("the duck gorps the bunny", scene, Cause(Duck, Move(Rabbit, "bend"))),

  ("the bunny florps", scene, Move(Rabbit, "lift"))

]


######
# Evaluation

ASSERTS = []
def setup_asserts():
  global ASSERTS
  ASSERTS = []

def _assert(expr, msg, raise_on_fail=False):
  """Lazy assert."""
  ASSERTS.append((expr, msg, raise_on_fail))

def teardown_asserts():
  n, successes = len(ASSERTS), 0
  for expr, msg, raise_on_fail in ASSERTS:
    try:
      assert expr
    except AssertionError:
      print("%sFAIL%s\t" % (Fore.RED, Style.RESET_ALL), msg)
      if raise_on_fail:
        raise
    else:
      successes += 1
      print("%s OK %s\t" % (Fore.GREEN, Style.RESET_ALL), msg)

  return successes / n if n > 0 else 0


def plot_distribution(distribution, name, out_dir=None, k=5, xlabel=None, title=None):
  """
  Save a bar plot of the given distribution.
  """
  if out_dir is None:
    out_dir = Path(".")

  support = sorted(distribution.keys(), key=lambda k: distribution[k], reverse=True)

  with (out_dir / ("%s.csv" % name)).open("w") as csv_f:
    for key in support:
      csv_f.write("%s,%f\n" % (key, distribution[key]))

  # Trim support for plot.
  if k is not None:
    support = support[:k]

  xs = np.arange(len(support))
  fig = plt.figure(figsize=(10, 8))
  plt.bar(xs, [distribution[support] for support in support])
  print([distribution[support] for support in support])
  plt.xticks(xs, list(map(str, support)), rotation="vertical")
  plt.ylabel("Probability mass")
  if xlabel is not None:
    plt.xlabel(xlabel)
  if title is not None:
    plt.title(title)

  plt.tight_layout()

  path = out_dir / ("%s.png" % name)
  L.info("Saving figure %s", path)
  fig.savefig(path)


def prep_example(learner, example):
  sentence, scene, answer = example
  sentence = sentence.split()
  model = Model(scene, learner.ontology)
  return sentence, model, answer


def zero_shot(learner, example, token):
  """
  Return zero-shot distributions p(syntax | example), p(syntax, meaning | example).
  """
  sentence, _, _ = prep_example(learner, example)
  syntaxes, joint_candidates = learner.predict_zero_shot(sentence)

  assert list(syntaxes.keys()) == [token]
  return syntaxes[token], joint_candidates[token].as_distribution()


def get_lexicon_distribution(learner, token):
  dist = Distribution()
  for entry in learner.lexicon._entries[token]:
    dist[entry.categ(), entry.semantics()] += entry.weight()
  return dist.normalize()


def compute_alternations(learner, constructions):
  cat_masses = learner.lexicon.total_category_masses()
  der_verb_cats = [der_cat for der_cat, _ in learner.lexicon._derived_categories.values()
                   if der_cat.base == learner.lexicon._start]
  table = pd.DataFrame(dtype=np.float, index=list(map(str, constructions)), columns=list(map(str, der_verb_cats)))
  for der_cat in der_verb_cats:
    for construction in constructions:
      query_cat = set_yield(construction, der_cat)
      table.loc[str(construction), str(der_cat)] = cat_masses[query_cat]
  table = table.subtract(table.min(axis=1), axis=0)
  table = table.div(table.max(axis=1), axis=0)
  return table


def eval_bootstrap_example(learner, example, token, expected_category,
                           asserts=True, extra=None):
  """
  Validate that verb bootstrapping using a derived category executes
  successfully, and update the model afterwards.
  """
  p_syn, p_syn_meaning = zero_shot(learner, example, token)

  plot_distribution(p_syn, "zeroshot.syntax.%s" % token)
  plot_distribution(p_syn_meaning, "zeroshot.joint.%s" % token)

  top_cat, top_expr = p_syn_meaning.argmax()
  if asserts and expected_category is not None:
    cat_yield = get_yield(top_cat)
    _assert(cat_yield == expected_category,
            "0-shot '%s' top candidate has yield %s: %s"
            % (token, expected_category, cat_yield))
    _assert(expected_category.source_name in str(top_expr),
            "0-shot invention %s is in '%s' top candidate expr: %s" %
            (expected_category.source_name, token, str(top_expr)))
  if extra is not None:
    extra(token, p_syn, p_syn_meaning)


def eval_oneshot_example(learner, example, token, expected_category,
                         asserts=True, extra=None):
  sentence, model, answer = prep_example(learner, example)
  learner.update_with_example(sentence, model, answer)

  # Fetch validated posterior from lexicon.
  posterior = get_lexicon_distribution(learner, token)
  plot_distribution(posterior, "oneshot.%s" % token)

  if asserts and expected_category is not None:
    top_cat, top_expr = posterior.argmax()
    cat_yield = get_yield(top_cat)
    _assert(cat_yield == expected_category,
            "1-shot '%s' top candidate has yield %s: %s"
            % (token, expected_category, cat_yield))
    _assert(expected_category.source_name in str(top_expr),
            "1-shot invention %s is in '%s' 1-shot top candidate expr: %s" %
            (expected_category.source_name, token, str(top_expr)))
  if extra is not None:
    extra(token, None, posterior)


def eval_model(compress=True, bootstrap=True, **learner_kwargs):
  L.info("Building model.")

  lexicon = initial_lexicon.clone()
  compressor = Compressor(lexicon.ontology, **EC_kwargs) if compress else None
  learner = WordLearner(lexicon, compressor, bootstrap=bootstrap,
                        **learner_kwargs)

  if compress:
    learner.compress_lexicon()

  # transitive verb -> causative semantics
  eval_bootstrap_example(learner, examples[0], "gorps", None)
  eval_oneshot_example(learner, examples[0], "gorps", None)

  # intransitive verb -> non-causative semantics
  eval_bootstrap_example(learner, examples[1], "florps", None)
  eval_oneshot_example(learner, examples[1], "florps", None)

  # # Run initial weight updates.
  # for example in examples:
  #   sentence, model, answer = prep_example(learner, example)
  #   learner.update_with_example(sentence, model, answer)


if __name__ == "__main__":
  eval_model(compress=True, learning_rate=0.1, bootstrap_alpha=0.9)
