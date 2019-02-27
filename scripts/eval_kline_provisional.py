"""
Kline evaluation with provisional lexicon given in the CogSci2019 submission.
"""


from argparse import ArgumentParser
from copy import copy
import json
import logging
import math
from pathlib import Path
import pprint
import random
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
    try:
      if ontology.constants_dict[manner].type.name != "manner":
        raise ValueError()
    except:
      # lookup failed
      raise ValueError()

    self.agent = agent
    self.manner = manner

  def __hash__(self):
    return hash(("move", self.agent, self.manner))

  def __str__(self):
    return "Move(%s, %s)" % (self.agent, self.manner)

  __repr__ = __str__


class Sneeze(Action):
  def __init__(self, agent):
    if not isinstance(agent, frozendict):
      raise TypeError()

    self.agent = agent

  def __hash__(self):
    return hash(("sneeze", self.agent))

  def __str__(self):
    return "Sneeze(%s)" % self.agent

  __repr__ = __str__


class Become(Action):
  def __init__(self, agent, state):
    if not isinstance(agent, frozendict):
      raise ValueError()
    if not isinstance(state, str):
      raise ValueError()
    try:
      if ontology.constants_dict[state].type.name != "state":
        raise ValueError()
    except:
      # lookup failed
      raise ValueError()

    self.agent = agent
    self.state = state

  def __hash__(self):
    return hash(("become", self.agent, self.state))

  def __str__(self):
    return "Become(%s -> %s)" % (self.agent, self.state)

  __repr__ = __str__


class Contact(Action):
  def __init__(self, a1, a2):
    if not isinstance(a1, frozendict) or not isinstance(a2, frozendict):
      raise TypeError()
    self.a1 = a1
    self.a2 = a2

  def __hash__(self):
    return hash(("contact", self.a1, self.a2))

  def __str__(self):
    return "Contact(%s -> %s)" % (self.a1, self.a2)

  __repr__ = __str__


INTENSIONAL_TYPES = [Action, Cause, Move, Become, Contact]
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
  types.new_function("contact", ("agent", "agent", "action"), Contact),
  types.new_function("sneeze", ("agent", "action"), Sneeze),

  types.new_function("unique", (("agent", "boolean"), "agent"), fn_unique),
  # types.new_function("nott", ("action", "action"), lambda a: NegatedAction(a)),
  types.new_function("not", ("boolean", "boolean"), fn_not),

  types.new_function("female", ("agent", "boolean"), lambda a: a["female"]),
  types.new_function("toy", ("agent", "boolean"), lambda a: a["type"] == "toy"),
  types.new_function("shade", ("agent", "boolean"), lambda a: a["type"] == "shade"),
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

  def evaluate(self, expr, no_lookup=False):
    if isinstance(expr, l.NegatedExpression):
      inner = self.evaluate(expr.term, no_lookup=no_lookup)
      if no_lookup:
        return NegatedAction(inner)
      if isinstance(inner, bool):
        return not inner
      return None

    ret = super().evaluate(expr)
    if no_lookup:
      return ret

    ret = super().evaluate(expr)
    if type(ret) in self.intensional_types:
      ret = ret in self.intensional_referents
    elif isinstance(ret, ComposedAction):
      ret = all(action in self.intensional_referents for action in ret.actions)
    elif isinstance(ret, NegatedAction):
      ret = not self.evaluate(ret.action)
    elif ret is not None and not isinstance(ret, bool):
      ret = ret in self.scene.objects

    return ret

######
# Lexicon.

initial_lexicon = Lexicon.fromstring(r"""
  :- S:N

  push => S\N/N {\y x.cause(x,move(y,bend))} <0.3>
  activate => S\N/N {\y x.cause(x,become(y,active))} <0.5>
  touch => S\N/N {\y x.contact(x,y)} <0.5>
  go => S\N {\x.move(x,bend)} <0.4>
  sneeze => S\N {\x.sneeze(x)} <0.4>
  the => N/N {\x.unique(x)} <1.1>
  girl => N {\x.female(x)} <0.6>
  toy => N {\x.toy(x)} <0.8>
  toy => N {\x.toy(x)} <0.3>
  doesn't => (S\N)/(S\N) {\a.not(a)} <0.5>
""", ontology, include_semantics=True)

######
# Evaluation data.
L.info("Preparing evaluation data.")

Ricky = frozendict(type="person", female=False)
Sarah = frozendict(type="person", female=True)
Wanda = frozendict(type="person", female=True)
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

  def __repr__(self):
    return "Scene%s{\n\tobjects=%s,\n\tevents=%s}" % \
      ("<%s>" % self.name if self.name else "",
       pprint.pformat(self.objects), pprint.pformat(self.events))

######

test_2afc_examples = [

  (("the girl gorps the toy",
    Scene([Sarah, Toy], [Cause(Sarah, Become(Toy, "active")), Contact(Sarah, Toy), Move(Sarah, "lift")],
          name="scene1"),
    Scene([Sarah, Toy], [Become(Toy, "active"), Move(Sarah, "lift")], name="scene2")),
   0),

  (("the girl doesn't gorps the toy",
    Scene([Sarah, Toy], [Cause(Sarah, Become(Toy, "active")), Contact(Sarah, Toy)]),
    Scene([Sarah, Toy], [Become(Toy, "active"), Move(Sarah, "lift")])),
   1),

  # (("the boy gorps the toy",
  #   Scene([Ricky, Donut], [Cause(Ricky, Become(Donut, "active")), Contact(Ricky, Donut)]),
  #   Scene([Ricky, Donut], [Become(Donut, "active"), Move(Ricky, "lift")])),
  #  0),

  # (("the boy doesn't gorps the toy",
  #   Scene([Ricky, Donut], [Cause(Ricky, Become(Donut, "active")), Contact(Ricky, Donut)]),
  #   Scene([Ricky, Donut], [Become(Donut, "active"), Move(Ricky, "lift")])),
  #  1),

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
  sentence, _, scene = example
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

  learner_kwargs["limit_induction"] = True
  pprint.pprint(learner_kwargs)

  lexical_limit = learner_kwargs.pop("lexical_limit", 1000)
  learner_kwargs["prune_entries"] = lexical_limit
  learner_kwargs["zero_shot_limit"] = lexical_limit

  default_weight = learner_kwargs.pop("weight_init")
  lexicon = initial_lexicon.clone()

  learner = WordLearner(lexicon, compressor=None, bootstrap=bootstrap,
                        **learner_kwargs)

  ###########

  for example, expected_idx in test_2afc_examples:
    sentence = example[0]
    print("------ ", sentence)
    eval_2afc_zeroshot(learner, example, expected_idx)



if __name__ == "__main__":
  hparams = [
    ("learning_rate", False, 1.0, 1.0, 1.0),
    ("bootstrap_alpha", False, 0.25, 0.25, 0.25),
    ("beta", False, 1.0, 1.0, 1.0),
    ("negative_samples", False, 7, 7, 7),
    ("total_negative_mass", False, 0.1, 0.1, 0.1),
    ("syntax_prior_smooth", True, 1e-3, 1e-3, 1e-3),
    ("meaning_prior_smooth", True, 1e-3, 1e-3, 1e-3),
    ("weight_init", True, 1e-2, 1e-2, 1e-2),
    ("lexical_limit", False, 1, 10, 3),
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
