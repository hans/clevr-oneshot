from argparse import ArgumentParser
from copy import copy
import logging
import math
from pathlib import Path
from pprint import pprint
import sys
from traceback import print_exc

from colorama import Fore, Style
from frozendict import frozendict
from nltk.sem import logic as l
import numpy as np

from clevros.lexicon import Lexicon, get_yield, set_yield
from clevros.logic import TypeSystem, Ontology, is_negation
from clevros.model import Model
from clevros.primitives import *
from clevros.util import Distribution
from clevros.word_learner import WordLearner

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


######
# Ontology.

class Cause(Action):
  def __init__(self, agent, behavior):
    if not isinstance(agent, frozendict):
      raise TypeError()
    if not isinstance(behavior, Action):
      raise TypeError()
    self.agent = agent
    self.behavior = behavior

  @property
  def entailed_actions(self):
    return [self.behavior]

  def __hash__(self):
    return hash((self.agent, self.behavior))

  def __str__(self):
    return "Cause(%s, %s)" % (self.agent, self.behavior)

  __repr__ = __str__


class Move(Action):
  def __init__(self, agent, manner):
    if not isinstance(agent, frozendict):
      raise TypeError()
    if not isinstance(manner, str):
      raise TypeError()
    self.agent = agent
    self.manner = manner

  def __hash__(self):
    return hash(("move", self.agent, self.manner))

  def __str__(self):
    return "Move(%s, %s)" % (self.agent, self.manner)

  __repr__ = __str__


class Become(Action):
  def __init__(self, agent, state):
    if not isinstance(agent, frozendict):
      raise ValueError()
    if not isinstance(state, str):
      raise ValueError()
    self.agent = agent
    self.state = state

  def __hash__(self):
    return hash(("become", self.agent, self.state))

  def __str__(self):
    return "Become(%s -> %s)" % (self.agent, self.state)

  __repr__ = __str__


INTENSIONAL_TYPES = [Action, Cause, Move, Become]
"""
Logical types which, when evaluated on a scene, should be compared to a stored
collection of intentional referents. (In other words, these types have truth
values which are determined by the intentional referent collection.)
"""

types = TypeSystem(["agent", "action", "manner", "state", "boolean"])

functions = [
  types.new_function("cause", ("agent", "action", "action"), Cause),
  types.new_function("move", ("agent", "manner", "action"), Move),
  types.new_function("become", ("agent", "state", "action"), Become),

  types.new_function("unique", (("agent", "boolean"), "agent"), fn_unique),
  types.new_function("nott", ("action", "action"), lambda a: NegatedAction(a)),
  types.new_function("not", ("boolean", "boolean"), lambda a: not a),

  types.new_function("female", ("agent", "boolean"), lambda a: a["female"]),
  types.new_function("toy", ("agent", "boolean"), lambda a: a["type"] == "toy"),
  types.new_function("shade", ("agent", "boolean"), lambda a: a["type"] == "shade"),
  types.new_function("shade", ("agent", "boolean"), lambda a: a["shape"] == "donut"),
]

constants = [
  types.new_constant("bend", "manner"),
  types.new_constant("lift", "manner"),
  types.new_constant("tilt", "manner"),

  types.new_constant("clean", "state"),
  types.new_constant("dirty", "state"),
  types.new_constant("active", "state"),
  types.new_constant("inactive", "state"),
]

ontology = Ontology(types, functions, constants)

######
# Logical evaluation routine.

class IntensionalModel(Model):
  """
  Logical model which supports scenes with intensional references.
  """

  def __init__(self, scene, ontology,
               intensional_types=None, intensional_referents=None):
    super().__init__(scene, ontology)
    # TODO support subclasses?
    self.intensional_types = set(intensional_types or [])

    try:
      intensional_referents = intensional_referents or scene.events
    except AttributeError:
      intensional_referents = [Action]
    for ref in intensional_referents:
      assert type(ref) in self.intensional_types, \
          "Intensional referent %s is not one of prescribed intensional types" % ref
    self.intensional_referents = set(intensional_referents or [])

  def evaluate(self, expr, v=False):
    if isinstance(expr, l.NegatedExpression):
      inner = self.evaluate(expr.term)
      if isinstance(inner, bool):
        return not inner
      return None

    ret = super().evaluate(expr)
    if type(ret) in self.intensional_types:
      return ret in self.intensional_referents
    if isinstance(ret, ComposedAction):
      return all(action in self.intensional_referents for action in ret.actions)
    if isinstance(ret, NegatedAction):
      return not self.evaluate(ret.action)
    if not isinstance(ret, bool):
      ret = ret in self.scene.objects
    return ret

######
# Lexicon.

initial_lexicon = Lexicon.fromstring(r"""
  :- S:N

  the => N/N {\x.unique(x)}
  an => N/N {\x.x}
  she => N {\x.female(x)}
  shade => N {\x.shade(x)}

  doesn't => (S\N)/(S\N) {\a.not(a)}

  bends => S\N/N {\p a.cause(a,move(p,bend))}
  lifts => S\N/N {\p a.cause(a,move(p,lift))}
  tilts => S\N/N {\p a.cause(a,move(p,tilt))}
  cleans => S\N/N {\p a.cause(a,become(p,clean))}
  dirties => S\N/N {\p a.cause(a,become(p,dirty))}
  switches => S\N/N {\p a.cause(a,become(p,active))}

  ducks => S\N {\a.move(a,bend)}
  jumps => S\N {\a.move(a,lift)}
  spins => S\N {\a.move(a,tilt)}
""", ontology, include_semantics=True)

######
# Evaluation data.
L.info("Preparing evaluation data.")

Ricky = frozendict(type="person", female=False)
Sarah = frozendict(type="person", female=True)
Toy = frozendict(type="toy")
Donut = frozendict(type="toy", shape="donut")
Wand = frozendict(type="toy", shape="rod")
Pendulum = frozendict(type="toy", shape="rod")
Globe = frozendict(type="toy", shape="sphere")
Shade = frozendict(type="shade")

class Scene(object):
  def __init__(self, objects, events, name=None, event_closure=False):
    """
    Args:
      objects:
      events:
      name:
      event_closure: If `True`, replace `events` with its closure under
        entailment.
    """
    self.objects = objects
    self.name = name

    events = set(events)
    if event_closure:
      while True:
        new_events = copy(events)
        for event in events:
          new_events |= set(event.entailed_actions)

        if events == new_events:
          # Closure is complete.
          break
        events = new_events
    self.events = events

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

  def __str__(self):
    if self.name: return "Scene<%s>" % self.name
    else: return super().__str__()

######

training_examples = [
  ("the toy", Scene([Sarah, Toy], [])),
  ("the toy", Scene([Ricky, Toy], [])),
  ("she florps the toy", Scene([Ricky, Sarah, Toy], [Cause(Sarah, Become(Toy, "active"))])),
  ("she florps the donut", Scene([Ricky, Sarah, Toy], [Cause(Sarah, Become(Toy, "active"))])),
  ("the donut", Scene([Ricky, Donut], [])),
  ("the donut", Scene([Sarah, Donut], [])),
  ("he florps the donut", Scene([Ricky, Sarah, Toy], [Cause(Ricky, Become(Toy, "active"))])),

  ("the toy", Scene([Sarah, Toy], [])),
  ("the toy", Scene([Ricky, Toy], [])),
  ("she florps the toy", Scene([Ricky, Sarah, Toy], [Cause(Sarah, Become(Toy, "active"))])),
  ("the toy", Scene([Sarah, Toy], [])),
  ("the toy", Scene([Ricky, Toy], [])),
  ("she florps the toy", Scene([Ricky, Sarah, Toy], [Cause(Sarah, Become(Toy, "active"))])),

]

test_2afc_examples = [

  (("she gorps the toy",
    Scene([Sarah, Toy], [Cause(Sarah, Become(Toy, "active"))]),
    Scene([Sarah, Toy], [Become(Toy, "active")])),
   0),

  # (("she doesn't gorps the toy",
  #   Scene([Sarah, Toy], [Cause(Sarah, Become(Toy, "active"))]),
  #   Scene([Sarah, Toy], [Become(Toy, "active")])),
  #  1),

  # (("she gorps the toy",
  #   Scene([Sarah, Donut], [Cause(Sarah, Become(Donut, "active"))]),
  #   Scene([Sarah, Donut], [Become(Donut, "active")])),
  #  0),

  # (("she gorps the toy",
  #   Scene([Sarah, Wand], [Cause(Sarah, Become(Wand, "active"))]),
  #   Scene([Sarah, Wand], [Become(Wand, "active")])),
  #  0),

  # (("she gorps the toy",
  #   Scene([Sarah, Pendulum], [Cause(Sarah, Move(Pendulum, "tilt"))]),
  #   Scene([Sarah, Pendulum], [Move(Pendulum, "tilt")])),
  #  0),

  # (("she gorps the toy",
  #   Scene([Sarah, Globe], [Cause(Sarah, Become(Globe, "active"))]),
  #   Scene([Sarah, Globe], [Become(Globe, "active")])),
  #  0),

  # (("she gorps the shade",
  #   Scene([Sarah, Shade], [Cause(Sarah, Move(Shade, "lift"))]),
  #   Scene([Sarah, Shade], [Move(Shade, "lift")])),
  #  0),

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


def prep_example(learner, example):
  sentence, scene = example
  sentence = sentence.split()
  model = IntensionalModel(scene, learner.ontology,
                           intensional_types=INTENSIONAL_TYPES)
  return sentence, model, None


def prep_example_2afc(learner, example):
  sentence, scene1, scene2 = example
  sentence = sentence.split()
  model1 = IntensionalModel(scene1, learner.ontology,
                            intensional_types=INTENSIONAL_TYPES)
  model2 = IntensionalModel(scene2, learner.ontology,
                            intensional_types=INTENSIONAL_TYPES)
  return sentence, model1, model2


def zero_shot(learner, example, token, prep_example_fn=prep_example):
  """
  Return zero-shot distributions p(syntax | example), p(syntax, meaning | example).
  """
  sentence, _, _ = prep_example_fn(learner, example)
  syntaxes, joint_candidates = learner.predict_zero_shot_tokens(sentence)

  assert list(syntaxes.keys()) == [token]
  return syntaxes[token], joint_candidates[token].as_distribution()


def get_lexicon_distribution(learner, token):
  dist = Distribution()
  for entry in learner.lexicon._entries[token]:
    dist[entry.categ(), entry.semantics()] += entry.weight()
  return dist.normalize()


def eval_bootstrap_example(learner, example, token, expected_category,
                           bootstrap, asserts=True, extra=None):
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


# def eval_oneshot_example(learner, example, token, expected_category,
#                          asserts=True, extra=None):
#   sentence, model, answer = prep_example(learner, example)
#   learner.update_with_distant(sentence, model, answer)

#   # Fetch validated posterior from lexicon.
#   posterior = get_lexicon_distribution(learner, token)
#   plot_distribution(posterior, "oneshot.%s" % token)

#   if asserts and expected_category is not None:
#     top_cat, top_expr = posterior.argmax()
#     cat_yield = get_yield(top_cat)
#     _assert(cat_yield == expected_category,
#             "1-shot '%s' top candidate has yield %s: %s"
#             % (token, expected_category, cat_yield))
#     _assert(expected_category.source_name in str(top_expr),

#             "1-shot invention %s is in '%s' 1-shot top candidate expr: %s" %
#             (expected_category.source_name, token, str(top_expr)))
#   if extra is not None:
#     extra(token, None, posterior)


def eval_2afc_zeroshot(learner, example, expected_idx, asserts=True):
  """
  Evaluate a zero-shot 2AFC selection on the two scenes specified by `model1`
  and `model2`.
  """
  sentence, model1, model2 = prep_example_2afc(learner, example)
  posterior = learner.predict_zero_shot_2afc(sentence, model1, model2)
  print(posterior)
  assert expected_idx in [0, 1]

  if asserts:
    _assert(abs(sum(posterior.values()) - 1) < 1e-7,
            "2AFC posterior is a valid probability distribution")

    top_model = posterior.argmax()
    _assert(top_model.scene == [model1, model2][expected_idx].scene,
            "Top model is expected for \"%s\"" % " ".join(sentence))


def eval_model(bootstrap=True, **learner_kwargs):
  L.info("Building model.")
  pprint(learner_kwargs)

  default_weight = learner_kwargs.pop("weight_init")
  lexicon = initial_lexicon.clone()

  learner = WordLearner(lexicon, compressor=None, bootstrap=bootstrap,
                        **learner_kwargs)

  ###########

  for example in training_examples:
    sentence, model, _ = prep_example(learner, example)
    print("------ ", " ".join(sentence))
    learner.update_with_cross_situational(sentence, model)
    learner.lexicon.debug_print()

  for example, expected_idx in test_2afc_examples:
    sentence = example[0]
    print("------ ", sentence)
    eval_2afc_zeroshot(learner, example, expected_idx)



if __name__ == "__main__":
  hparams = [
    ("learning_rate", False, 0.1, 0.5, 0.3),
    ("bootstrap_alpha", False, 0.0, 1.0, 0.25),
    ("beta", False, 0.1, 1.1, 0.5),
    ("negative_samples", False, 5, 20, 7),
    ("total_negative_mass", False, 0.1, 1.0, 0.1),
    ("syntax_prior_smooth", True, 1e-5, 1e-1, 1e-3),
    ("meaning_prior_smooth", True, 1e-9, 1e-2, 1e-3),
    ("weight_init", True, 1e-4, 1e-1, 1e-2),
    ("prune_weight", False, -0.1, 0.1, 1e-4),
  ]

  p = ArgumentParser()
  p.add_argument("-o", "--out_dir", default=".", type=Path)
  p.add_argument("-m", "--mode", choices=["search", "eval"], default="eval")
  p.add_argument("-s", "--search_file", default="search.log", type=str)
  for hparam, _, _, _, default in hparams:
    p.add_argument("--%s" % hparam, default=default, type=type(default))

  args = p.parse_args()

  def sample_hparam(hparam):
    name, log_scale, low, high, _ = hparam
    if log_scale:
      low, high = math.log(low, 10), math.log(high, 10)
    val = np.random.uniform(low, high)
    if log_scale:
      val = math.pow(10, val)
    if isinstance(low, int):
      val = int(round(val))
    return val

  if args.mode == "search":
    search_f = open(args.search_file, "a")
    search_f.write("success_ratio\t")
    search_f.write("\t".join(hparam for hparam, _, _, _, _ in hparams))
    search_f.write("\n")
  else:
    search_f = sys.stderr

  while True:
    sampled_hparams = {
      hparam[0]: (sample_hparam(hparam) if args.mode == "search"
                  else getattr(args, hparam[0]))
      for hparam in hparams
    }

    setup_asserts()
    try:
      eval_model(**sampled_hparams)
      result = teardown_asserts()
    except KeyboardInterrupt:
      sys.exit(1)
    except Exception as e:
      print_exc()
      search_f.write("0\t")
    else:
      search_f.write("%.3f\t" % result)

    search_f.write("\t".join("%g" % sampled_hparams[hparam]
                            for hparam, _, _, _, _ in hparams))
    search_f.write("\n")

    search_f.flush()

    if args.mode == "eval":
      break
