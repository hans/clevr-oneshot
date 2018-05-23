"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

from argparse import ArgumentParser
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
  types.new_function("cmp_pos", ("ax", "manner", "v", "obj", "num"), fn_cmp_pos),
  types.new_function("ltzero", ("num", "boolean"), fn_ltzero),
  types.new_function("and_", ("boolean", "boolean", "boolean"), fn_and),
  types.new_function("constraint", ("boolean", "manner"), lambda fn: Constraint(fn)),
  types.new_function("e", ("v",), Event()),

  types.new_function("ax_x", ("ax",), fn_ax_x),
  types.new_function("ax_y", ("ax",), fn_ax_y),
  types.new_function("ax_z", ("ax",), fn_ax_z),

  types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),

  types.new_function("cube", ("obj", "boolean"), fn_cube),
  types.new_function("sphere", ("obj", "boolean"), fn_sphere),
  types.new_function("donut", ("obj", "boolean"), fn_donut),
  types.new_function("pyramid", ("obj", "boolean"), fn_pyramid),
  types.new_function("hose", ("obj", "boolean"), fn_hose),
  types.new_function("cylinder", ("obj", "boolean"), fn_cylinder),

  types.new_function("letter", ("obj", "boolean"), lambda x: x["shape"] == "letter"),
  types.new_function("package", ("obj", "boolean"), lambda x: x["shape"] == "package"),

  types.new_function("male", ("obj", "boolean"), lambda x: not x["female"]),
  types.new_function("female", ("obj", "boolean"), lambda x: x["female"]),

  types.new_function("young", ("obj", "boolean"), lambda x: x["age"] < 20),
  types.new_function("old", ("obj", "boolean"), lambda x: x["age"] > 20),

  types.new_function("object", (types.ANY_TYPE, "boolean"), fn_object),
  types.new_function("agent", ("obj", "boolean"), lambda x: x["agent"]),

  # TODO types don't make sense here
  types.new_function("move", ("obj", ("obj", "boolean"), "manner", "action"), lambda obj, dest, manner: Move(obj, dest, manner)),
  types.new_function("cause_possession", ("obj", "obj", "action"), lambda agent, obj: CausePossession(agent, obj)),
  types.new_function("transfer", ("obj", "obj", "manner", "action"), lambda obj, agent, dist: Transfer(obj, agent, dist)),

  types.new_function("do_", ("action", "action", "action"), lambda a1, a2: a1 + a2),
]

constants = [types.new_constant("any", "manner"),
             types.new_constant("far", "manner"),
             types.new_constant("near", "manner"),
             types.new_constant("slow", "manner"),
             types.new_constant("fast", "manner"),
             types.new_constant("pos", "manner"),
             types.new_constant("neg", "manner")]

ontology = Ontology(types, functions, constants, variable_weight=0.1)

#####################

types_levin = TypeSystem(["obj", "boolean", "manner", "action", "set", "str"])

functions_levin = [
  # Logical operation
  types_levin.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),
  types_levin.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
  types_levin.new_function("eq", ("?", "?", "boolean"), lambda a, b: a == b),

  # Ops on objects
  types_levin.new_function("book", ("obj", "boolean"), lambda x: x["type"] == "book"),
  # types_levin.new_function("apple", ("obj", "boolean"), lambda x: x["type"] == "apple"),
  # types_levin.new_function("cookie", ("obj", "boolean"), lambda x: x["type"] == "cookie"),
  types_levin.new_function("water", ("obj", "boolean"), lambda x: x["substance"] == "water"),
  types_levin.new_function("paint", ("obj", "boolean"), lambda x: x["substance"] == "paint"),
  types_levin.new_function("orientation", ("obj", "manner"), lambda x: x["orientation"]),
  types_levin.new_function("liquid", ("obj", "boolean"), lambda x: x["state"] == "liquid"),
  types_levin.new_function("full", ("obj", "boolean"), lambda x: x["full"]),

  # Ops on sets
  types_levin.new_function("characteristic", ("set", "str", "boolean"),
                           lambda s, fn: s.characteristic == fn),

  # Ops on events
  types_levin.new_function("e", ("v",), Event()),
  types_levin.new_function("direction", ("v", "manner"), lambda e: e.direction),
  types_levin.new_function("result", ("v", "obj"), lambda e: e.result),

  # Actions
  types_levin.new_function("put", ("v", "obj", "manner", "action"), Put),
  types_levin.new_function("constraint", ("boolean", "manner"), Constraint), # TODO funky type
  types_levin.new_function("addc", ("manner", "manner", "manner"), lambda c1, c2: Constraint(c1, c2)),
  types_levin.new_function("join", ("action", "boolean", "action"), ActAndEntail),
]


constants_levin = [
  types_levin.new_constant("vertical", "manner"),
  types_levin.new_constant("horizontal", "manner"),
  types_levin.new_constant("up", "manner"),
  types_levin.new_constant("down", "manner"),

  types_levin.new_constant("apple", "str"),
  types_levin.new_constant("cookie", "str"),
]

ontology_levin = Ontology(types_levin, functions_levin, constants_levin)

#####################


semantics = True

lex_vol = Lexicon.fromstring(r"""
  :- S, PP, N

  cube => N {\x.and_(object(x),cube(x))}
  sphere => N {\x.and_(object(x),sphere(x))}
  donut => N {\x.and_(object(x),donut(x))}
  hose => N {\x.and_(object(x),hose(x))}
  cylinder => N {\x.and_(object(x),cylinder(x))}
  pyramid => N {\x.and_(object(x),pyramid(x))}

  the => N/N {\x.unique(x)}

  below => PP/N {\a.constraint(ltzero(cmp_pos(ax_z,pos,e,a)))}
  right_of => PP/N {\a.constraint(ltzero(cmp_pos(ax_x,neg,e,a)))}
  in_front_of => PP/N {\a.constraint(ltzero(cmp_pos(ax_y,neg,e,a)))}

  put => S/N/PP {\b a.move(a,b,slow)}
  drop => S/N/PP {\b a.move(a,b,fast)}
  """, ontology=ontology, include_semantics=semantics)

lex_voo = Lexicon.fromstring(r"""
  :- S, N

  woman => N {\x.and_(agent(x),female(x))}
  man => N {\x.and_(agent(x),male(x))}

  letter => N {\x.letter(x)}
  ball => N {\x.sphere(x)}
  package => N {\x.package(x)}

  the => N/N {\x.unique(x)}

  give => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,any))}
  send => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,far))}
  hand => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,near))}
  """, ontology, include_semantics=True)


lex_levin = Lexicon.fromstring(r"""
  :- S, N, PP

  the => N/N {\x.unique(x)}

  book => N {\x.book(x)}
  water => N {\x.water(x)}
  paint => N {\x.paint(x)}

  apples => N {\x.characteristic(x,apple)}
  cookies => N {\x.characteristic(x,cookie)}

  put => S/N/PP {\d o.put(e,o,d)}
  set => S/N/PP {\d o.put(e,o,d)}

  # "hang the picture on the wall"
  hang => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(orientation(result(e)),vertical))))}
  lay => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(orientation(result(e)),horizontal))))}

  drop => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(direction(e),down))))}
  hoist => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(direction(e),up))))}

  pour => S/N/PP {\d o.join(put(e,o,d),liquid(result(e)))}
  spill => S/N/PP {\d o.join(put(e,o,d),liquid(result(e)))}

  # "spray the wall with paint"
  # spray => S/N/PP {\d o.and_(put(e,o,addc(d,constraint(not(full(result(e)))))))}
  # load => S/N/PP {\d o.put(e,o,addc(d,constraint(not(full(result(e))))))}

  # "fill the jar with cookies"
  fill => S/N/PP {\o d.put(e,o,addc(d,constraint(full(result(e)))))}
  stuff => S/N/PP {\o d.put(e,o,addc(d,constraint(full(result(e)))))}
  """, ontology_levin, include_semantics=True)


scene = {
  "objects": [
    frozendict({"female": True, "agent": True, "age": 18, "shape": "person"}),
    frozendict({"shape": "letter"}),
    frozendict({"shape": "package"}),
    frozendict({"female": False, "agent": True, "age": 5, "shape": "person"}),
    frozendict({"shape": "sphere"}),
  ]
}
scene2 = {
  "objects": [
    frozendict({"female": True, "agent": True, "age": 9, "shape": "person"}),
    frozendict({"shape": "sphere"}),
  ]
}
scene_vol = \
  {'directions': {'above': [0.0, 0.0, 1.0],
                  'behind': [-0.754490315914154, 0.6563112735748291, 0.0],
                  'below': [-0.0, -0.0, -1.0],
                  'front': [0.754490315914154, -0.6563112735748291, -0.0],
                  'left': [-0.6563112735748291, -0.7544902563095093, 0.0],
                  'right': [0.6563112735748291, 0.7544902563095093, -0.0]},
 'image_filename': 'CLEVR_train_000002.png',
 'image_index': 2,
 'objects': [
             frozendict({
               '3d_coords': (2.1141371726989746,
                            1.0,
                            2),
              'color': 'yellow',
              'material': 'metal',
              'rotation': 308.49217566676606,
              'shape': 'sphere',
              'size': 'large'}),
            frozendict({
              '3d_coords': (0, 0, 0),
              'color': 'blue',
              'material': 'rubber',
              'pixel_coords': (188, 94, 12.699371337890625),
              'rotation': 82.51702981683107,
              'shape': 'pyramid',
              'size': 'large'}),
             frozendict({
               '3d_coords': (-2.3854215145111084,
                            0.0,
                            0.699999988079071),
              'color': 'blue',
              'material': 'rubber',
              'rotation': 82.51702981683107,
              'shape': 'cube',
              'size': 'large'})],
 'split': 'train'}


examples_vol = [
  ("place the sphere right_of the cube", scene_vol,
   Move(scene_vol["objects"][0],
        Constraint(-1 * (Event()["3d_coords"][0] - scene_vol["objects"][2]["3d_coords"][0]) < 0),
        "slow")),
]
examples_voo = [
    ("send the woman the package", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][2]),
                    Transfer(scene["objects"][2], scene["objects"][0], "far"))),
    ("send the boy the package", scene,
     ComposedAction(CausePossession(scene["objects"][3], scene["objects"][2]),
                    Transfer(scene["objects"][2], scene["objects"][3], "far"))),
    ("hand the kid the ball", scene2,
     ComposedAction(CausePossession(scene2["objects"][0], scene2["objects"][1]),
                    Transfer(scene2["objects"][1], scene2["objects"][0], "near"))),
    ("give the lady the ball", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][4]),
                    Transfer(scene["objects"][4], scene["objects"][0], "any"))),

    ("gorp the woman the letter", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][1]),
                    Transfer(scene["objects"][1], scene["objects"][0], "far"))),
]
examples_levin = [

]

def compress_lexicon(lex, compressor):
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

def main(args, lex, ontology, examples):
  if not args.no_compress:
    # Run compression on the initial lexicon.
    compressor = Compressor(ontology)
    lex = compress_lexicon(lex, compressor)

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
                                    sentence, ontology, model, answer,
                                    bootstrap=not args.no_bootstrap)

      if not args.no_compress:
        # Run compression on the augmented lexicon.
        lex = compress_lexicon(lex, compressor)

      # Attempt a new parameter update.
      weighted_results, _ = update_perceptron_distant(lex, sentence, model, answer)

    final_sem = weighted_results[0][0].label()[0].semantics()

    print(" ".join(sentence), len(weighted_results), final_sem)
    print("\t", model.evaluate(final_sem))


if __name__ == "__main__":
  p = ArgumentParser()

  # Model lesion options.
  p.add_argument("--no-compress", action="store_true")
  p.add_argument("--no-bootstrap", action="store_true")

  # main(p.parse_args(), lex_voo, ontology, examples_voo)
  # main(p.parse_args(), lex_vol, ontology, examples_vol)
  main(p.parse_args(), lex_levin, ontology_levin, examples_levin)
