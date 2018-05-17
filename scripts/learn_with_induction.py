"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

import inspect
import logging
import numbers
from pprint import pprint

from frozendict import frozendict
from nltk.ccg import chart
import numpy as np

from clevros.chart import WeightedCCGChartParser, DefaultRuleSet
from clevros.lexicon import Lexicon, Token, \
    augment_lexicon_distant, get_candidate_categories
from clevros.logic import Ontology, Function, TypeSystem
from clevros.model import Model
from clevros.perceptron import update_perceptron_distant
from clevros.compression import Compressor

import random
random.seed(4)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")
L = logging.getLogger(__name__)


#compression params:
EC_kwargs = {
  "topK": 1,
  "pseudoCounts": 1.0,
  "a": 1,
  "aic": -1.0,
  "structurePenalty": 0.001,
  "backend": "pypy",
  "CPUs": 1,
}


#####################

from clevros.primitives import *

types = TypeSystem(["obj", "num", "ax", "manner", "boolean", "action"])

functions = [
  # types.new_function("cmp_pos", ("ax", "obj", "obj", "num"), fn_cmp_pos),
  # types.new_function("ltzero", ("num", "boolean"), fn_ltzero),
  types.new_function("and_", ("boolean", "boolean", "boolean"), fn_and),

  types.new_function("ax_x", ("ax",), fn_ax_x),
  types.new_function("ax_y", ("ax",), fn_ax_y),
  types.new_function("ax_z", ("ax",), fn_ax_z),

  types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),

  # types.new_function("cube", ("obj", "boolean"), fn_cube),
  types.new_function("sphere", ("obj", "boolean"), fn_sphere),
  # types.new_function("donut", ("obj", "boolean"), fn_donut),
  # types.new_function("pyramid", ("obj", "boolean"), fn_pyramid),
  # types.new_function("hose", ("obj", "boolean"), fn_hose),
  # types.new_function("cylinder", ("obj", "boolean"), fn_cylinder),

  types.new_function("letter", ("obj", "boolean"), lambda x: x["shape"] == "letter"),
  types.new_function("package", ("obj", "boolean"), lambda x: x["shape"] == "package"),

  types.new_function("male", ("obj", "boolean"), lambda x: not x["female"]),
  types.new_function("female", ("obj", "boolean"), lambda x: x["female"]),

  types.new_function("young", ("obj", "boolean"), lambda x: x["age"] < 20),
  types.new_function("old", ("obj", "boolean"), lambda x: x["age"] > 20),

  types.new_function("object", (types.ANY_TYPE, "boolean"), fn_object),
  types.new_function("agent", ("obj", "boolean"), lambda x: x["agent"]),

  types.new_function("move", ("obj", ("obj", "boolean"), "manner", "action"), lambda obj, dest, manner: Move(a, b, manner)),
  types.new_function("cause_possession", ("obj", "obj", "action"), lambda agent, obj: CausePossession(agent, obj)),
  types.new_function("transfer", ("obj", "obj", "manner", "action"), lambda obj, agent, dist: Transfer(obj, agent, dist)),

  types.new_function("do_", ("action", "action", "action"), lambda a1, a2: a1 + a2),
]

constants = [types.new_constant("any", "manner"),
             types.new_constant("far", "manner"),
             types.new_constant("near", "manner"),
             types.new_constant("slow", "manner"),
             types.new_constant("fast", "manner")]

ontology = Ontology(types, functions, constants, variable_weight=0.1)

#####################


semantics = True

lex_vol = Lexicon.fromstring(r"""
  :- S, PP, Nd, N

  cube => N {\x.and_(object(x),cube(x))}
  sphere => N {\x.and_(object(x),sphere(x))}
  donut => N {\x.and_(object(x),donut(x))}
  hose => N {\x.and_(object(x),hose(x))}
  cylinder => N {\x.and_(object(x),cylinder(x))}
  pyramid => N {\x.and_(object(x),pyramid(x))}

  the => Nd/N {\x.unique(x)}

  below => PP/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  # behind =>  PP/Nd {\b.\a.ltzero(cmp_pos(ax_y,b,a))}
  # above => PP/Nd {\b.\a.ltzero(cmp_pos(ax_z,b,a))}
  # left_of => PP/Nd {\b.\a.ltzero(cmp_pos(ax_x,a,b))}
  right_of => PP/Nd {\b.\a.ltzero(cmp_pos(ax_x,b,a))}
  in_front_of => PP/Nd {\b.\a.ltzero(cmp_pos(ax_y,a,b))}

  put => S/Nd/PP {\a.\b.move(a,b,slow)}
  drop => S/Nd/PP {\a.\b.move(a,b,fast)}
  """, ontology=ontology, include_semantics=semantics)

lex = Lexicon.fromstring(r"""
  :- S, N

  woman => N {\x.and_(agent(x),female(x))}
  man => N {\x.and_(agent(x),male(x))}

  letter => N {letter}
  ball => N {sphere}
  package => N {package}

  the => N/N {\x.unique(x)}

  give => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,any))}
  give => S/N/N {\a x.transfer(x,a,any)}
  send => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,far))}
  send => S/N/N {\a x.transfer(x,a,far)}
  hand => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,near))}
  hand => S/N/N {\a x.transfer(x,a,near)}
  """, ontology, include_semantics=True)


scene = {
  "objects": [
    frozendict({"female": True, "agent": True, "age": 18, "shape": "person"}),
    frozendict({"shape": "letter"}),
    frozendict({"shape": "package"}),
    frozendict({"female": False, "agent": True, "age": 5, "shape": "person"}),
    frozendict({"shape": "sphere"}),
    # frozendict({"female": False, "agent": True, "age": 62, "shape": "person"}),
  ]
}
examples_vol = [
  ("place the donut right_of the cube", scene,
   Move(scene["objects"][1], scene["objects"][2], "slow")),
]
examples = [
    ("send the boy the package", scene,
     ComposedAction(CausePossession(scene["objects"][3], scene["objects"][2]),
                    Transfer(scene["objects"][2], scene["objects"][3], "far"))),
    ("give the lady the ball", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][4]),
                    Transfer(scene["objects"][4], scene["objects"][0], "any"))),

    ("gorp the woman the letter", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][2]),
                    Transfer(scene["objects"][2], scene["objects"][0], "far"))),
]

compressor = Compressor(ontology, **EC_kwargs)

def compress_lexicon(lex):
  # Run EC compression on the entries of the induced lexicon. This may create
  # new inventions, updating both the `ontology` and the provided `lex`.
  lex, affected_entries = compressor.make_inventions(lex)

  for invention_name, tokens in affected_entries.items():
    if invention_name in lex._derived_categories_by_source:
      continue

    affected_syntaxes = set(t.categ() for t in tokens)
    if len(affected_syntaxes) == 1:
      # Just one syntax is involved. Create a new derived category.
      L.debug("Creating new derived category for tokens %r", tokens)

      derived_name = lex.add_derived_category(tokens, source_name=invention_name)
      lex.propagate_derived_category(derived_name)

      L.info("Created and propagated derived category %s == %s -- %r",
             derived_name, lex._derived_categories[derived_name][0].base, tokens)

  return lex


#############

if __name__ == "__main__":
  # Run compression on the initial lexicon.
  lex = compress_lexicon(lex)

  for sentence, scene, answer in examples:
    print("\n\n")

    sentence = sentence.split()

    model = Model(scene, ontology)

    try:
      weighted_results, _ = update_perceptron_distant(lex, sentence, model, answer)
    except ValueError:
      # No parse succeeded -- attempt lexical induction.
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))

      query_tokens = [word for word in sentence if not lex._entries.get(word, [])]
      L.info("Novel words: %s", " ".join(query_tokens))
      query_token_syntaxes = get_candidate_categories(lex, query_tokens, sentence)

      # Augment the lexicon with all entries for novel words which yield the
      # correct answer to the sentence under some parse. Restrict the search by
      # the supported syntaxes for the novel words (`query_token_syntaxes`).
      lex = augment_lexicon_distant(lex, query_tokens, query_token_syntaxes,
                                    sentence, ontology, model, answer)

      # Run compression on the augmented lexicon.
      lex = compress_lexicon(lex)

      # Attempt a new parameter update.
      weighted_results, _ = update_perceptron_distant(lex, sentence, model, answer)

    final_sem = weighted_results[0][0].label()[0].semantics()

    print(" ".join(sentence), len(weighted_results), final_sem)
    print("\t", model.evaluate(final_sem))
