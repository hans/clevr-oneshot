from argparse import ArgumentParser
import logging
import math
from pathlib import Path

from colorama import Fore, Style
from frozendict import frozendict
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from clevros.compression import Compressor
from clevros.environments import levin
from clevros.lexicon import get_candidate_categories, predict_zero_shot, get_yield, set_yield
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
# Evaluation data.
L.info("Preparing evaluation data.")

scene = {
  "objects": [
    Object(type="jar"),
    Object(type="book"),
    Object(type="table"),
    Collection("cookie"),
    Object(type="water"),
  ]
}
objs = scene["objects"]

event = Event()

examples = [

  # Weight updates
  ("put the book on the table", scene, Put(event, objs[1], Constraint(event.result.contact(objs[2])))),
  ("fill the jar with the cookies", scene, Put(event, objs[3], Constraint(Contain(objs[0], event.result), event.patient.full))),

  # BOOTSTRAP
  ("place the book on the table", scene, Put(event, objs[1], Constraint(event.result.contact(objs[2])))),
  ("cover the table with the cookies", scene, Put(event, objs[3], Constraint(Contain(objs[2], event.result), event.patient.full))),

  # NOVEL FRAME -- learn that "fill" class alternates
  ("fill the jar", scene, Put(event, event.result, Constraint(Contain(objs[0], event.result), event.patient.full))),

  # Bootstrap on that novel frame
  ("stuff the jar", scene, Put(event, event.result, Constraint(Contain(objs[0], event.result), event.patient.full))),

  # Weight updates -- new frames
  ("drop the jar on the table", scene, Put(event, objs[0], Constraint(event.result.contact(objs[2]), event.direction.equals("down")))),
  ("raise the book onto the table", scene, Put(event, objs[1], Constraint(event.result.contact(objs[2]), event.direction.equals("up")))),
  ("spill the water on the table", scene, Put(event, objs[4], Constraint(event.result.contact(objs[2]), event.result.state.equals("liquid")))),
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


def plot_distribution(distribution, name, k=5, xlabel=None, title=None):
  """
  Save a bar plot of the given distribution.
  """
  support = sorted(distribution.keys(), key=lambda k: distribution[k], reverse=True)

  with (args.out_dir / ("%s.csv" % name)).open("w") as csv_f:
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

  path = args.out_dir / ("%s.png" % name)
  L.info("Saving figure %s", path)
  fig.savefig(path)


def prep_example(learner, example):
  sentence, scene, answer = example
  sentence = sentence.split()
  model = Model(scene, learner.ontology)
  return sentence, model, answer


def zero_shot(learner, example, token, bootstrap=True):
  """
  Return zero-shot distributions p(syntax | example), p(syntax, meaning | example).
  """
  sentence, _, _ = prep_example(learner, example)
  cand_categories = get_candidate_categories(learner.lexicon, [token], sentence)
  candidates, _, _ = predict_zero_shot(learner.lexicon, [token], cand_categories, sentence,
                                       learner.ontology, bootstrap=bootstrap)
  return cand_categories[token], candidates[token].as_distribution()


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
                           bootstrap, asserts=True, extra=None):
  """
  Validate that verb bootstrapping using a derived category executes
  successfully, and update the model afterwards.
  """
  p_syn, p_syn_meaning = zero_shot(learner, example, token, bootstrap=bootstrap)

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

  lexicon = levin.lexicon.clone()
  compressor = Compressor(lexicon.ontology, **EC_kwargs) if compress else None
  learner = WordLearner(lexicon, compressor, bootstrap=bootstrap,
                        **learner_kwargs)

  # Run compression.
  learner.compress_lexicon()

  try:
    PP_CONTACT_CATEGORY, _ = learner.lexicon._derived_categories["D0"]
    PUT_CATEGORY, _ = learner.lexicon._derived_categories["D1"]
    DROP_CATEGORY, _ = learner.lexicon._derived_categories["D2"]
    POUR_CATEGORY, _ = learner.lexicon._derived_categories["D3"]
    FILL_CATEGORY, _ = learner.lexicon._derived_categories["D4"]
    # sanity check
    _assert(str(PP_CONTACT_CATEGORY.base) == "PP", "PP contact derived cat has correct base")
    _assert(str(PUT_CATEGORY.base) == "S", "Put verb derived cat has correct base")
    _assert(str(DROP_CATEGORY.base) == "S", "Drop verb derived cat has correct base")
    _assert(str(FILL_CATEGORY.base) == "S", "Fill verb derived cat has correct base")
    _assert(str(POUR_CATEGORY.base) == "S", "Pour verb derived cat has correct base")
  except KeyError:
    _assert(False, "Derived categories not available", False)
    PP_CONTACT_CATEGORY = None
    PUT_CATEGORY = None
    DROP_CATEGORY = None
    FILL_CATEGORY = None
    POUR_CATEGORY = None

  # Constructions in which different derived verb cats appear
  locative_construction = learner.lexicon.parse_category("S/N/PP")
  locative_construction._arg = PP_CONTACT_CATEGORY
  constructions = [
    locative_construction,
    learner.lexicon.parse_category("S/N/PP"),
    learner.lexicon.parse_category("S/N"),
  ]

  ###########

  # Run initial weight updates.
  for example in examples[:2]:
    sentence, model, answer = prep_example(learner, example)
    learner.update_with_example(sentence, model, answer)

  # Zero-shot predictions with bootstrapping in known frames
  def make_extra(target):
    def extra_check(token, cand_cats, cand_joint):
      top_cat, top_expr = cand_joint.argmax()
      _assert(top_cat.arg() == target,
              "Top cat for %s should have first arg of type %s: %s" %
              (token, target, top_cat))
    return extra_check
  eval_bootstrap_example(learner, examples[2], "place", PUT_CATEGORY,
                         bootstrap=bootstrap, extra=make_extra(PP_CONTACT_CATEGORY))
  eval_oneshot_example(learner, examples[2], "place", PUT_CATEGORY,
                       extra=make_extra(PP_CONTACT_CATEGORY))
  print(compute_alternations(learner, constructions))

  eval_bootstrap_example(learner, examples[3], "cover", FILL_CATEGORY,
                         bootstrap=bootstrap, extra=make_extra(learner.lexicon.parse_category("PP")))
  eval_oneshot_example(learner, examples[3], "cover", FILL_CATEGORY,
                       extra=make_extra(learner.lexicon.parse_category("PP")))
  print(compute_alternations(learner, constructions))

  # Learn a novel frame for the fill class.
  eval_bootstrap_example(learner, examples[4], "fill", FILL_CATEGORY, bootstrap=bootstrap)
  eval_oneshot_example(learner, examples[4], "fill", FILL_CATEGORY)
  print(compute_alternations(learner, constructions))

  # Zero-shot predictions for the newly learned frame.
  eval_bootstrap_example(learner, examples[5], "stuff", FILL_CATEGORY, bootstrap=bootstrap)
  eval_oneshot_example(learner, examples[5], "stuff", FILL_CATEGORY)
  print(compute_alternations(learner, constructions))

  eval_oneshot_example(learner, examples[6], "drop", DROP_CATEGORY)
  eval_oneshot_example(learner, examples[7], "raise", DROP_CATEGORY)
  print(compute_alternations(learner, constructions))

  eval_oneshot_example(learner, examples[8], "spill", POUR_CATEGORY)

  ###########

  # Produce alternation table.
  table = compute_alternations(learner, constructions)
  print(table)
  table.to_csv(args.out_dir / "alternations.csv")
  plt.clf()
  sns.heatmap(table)
  plt.savefig(args.out_dir / "alternations.png")


if __name__ == "__main__":
  hparams = [
    ("learning_rate", False, 1.0, 10.0, 5.0),
    ("bootstrap_alpha", False, 0.1, 0.9, 0.25),
    ("beta", False, 0.5, 10.0, 3.0),
    ("negative_samples", False, 1, 10, 5),
    ("total_negative_mass", False, 0.1, 1.0, 0.1),
    ("syntax_prior_smooth", True, 1e-6, 0.1, 1e-3),
  ]

  p = ArgumentParser()
  p.add_argument("--out_dir", default=".", type=Path)
  p.add_argument("-m", "--mode", choices=["search", "eval"])
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
    return val

  with open("search.log", "a") as search_f:
    search_f.write("success_ratio\t")
    search_f.write("\t".join(hparam for hparam, _, _, _, _ in hparams))
    search_f.write("\n")

    while True:
      sampled_hparams = {
        hparam[0]: (sample_hparam(hparam) if args.mode == "search"
                    else getattr(args, hparam[0]))
        for hparam in hparams
      }

      setup_asserts()
      eval_model(**sampled_hparams)
      result = teardown_asserts()

      search_f.write("%.3f\t" % result)
      search_f.write("\t".join("%g" % sampled_hparams[hparam]
                              for hparam, _, _, _, _ in hparams))
      search_f.write("\n")

      search_f.flush()

      if args.mode == "eval":
        break
