"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

import inspect
import numbers
from pprint import pprint

from frozendict import frozendict
from nltk.ccg import chart
import numpy as np

from clevros.chart import WeightedCCGChartParser
from clevros.lexicon import Lexicon, Token, \
    augment_lexicon_distant, get_candidate_categories
from clevros.logic import Ontology, Function, TypeSystem
from clevros.model import Model
from clevros.perceptron import update_perceptron_batch
from clevros.ec_util import Compressor

import random
random.seed(4)

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


# Teeny subset of CLEVR dataset :)
scene = \
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


#####################


semantics = True

lex = Lexicon.fromstring(r"""
  :- S, PP, Nd, N

  cube => N {\x.and_(object(x),cube(x))}
  sphere => N {\x.and_(object(x),sphere(x))}
  donut => N {\x.and_(object(x),donut(x))}
  hose => N {\x.and_(object(x),hose(x))}
  cylinder => N {\x.and_(object(x),cylinder(x))}
  pyramid => N {\x.and_(object(x),pyramid(x))}

  the => Nd/N {\x.unique(x)}

  below => PP/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  behind =>  PP/Nd {\b.\a.ltzero(cmp_pos(ax_y,b,a))}
  above => PP/Nd {\b.\a.ltzero(cmp_pos(ax_z,b,a))}
  left_of => PP/Nd {\b.\a.ltzero(cmp_pos(ax_x,a,b))}
  right_of => PP/Nd {\b.\a.ltzero(cmp_pos(ax_x,b,a))}
  in_front_of => PP/Nd {\b.\a.ltzero(cmp_pos(ax_y,a,b))}

  # todo not the right syntactic category
  # Need to first untie syntactic / semantic arities.
  put => S/Nd/PP {\a.\b.move(a,b)}
  place => S/Nd/PP {\a.\b.move(a,b)}
  """, include_semantics=semantics)

lex = Lexicon.fromstring(r"""
  :- S, N

  woman => N {\x.and_(agent(x),female(x))}
  man => N {\x.and_(agent(x),male(x))}

  # letter => N {\x.letter(x)}
  # ball => N {\x.ball(x)}
  # package => N {\x.package(x)}

  cube => N {\x.and_(object(x),cube(x))}
  sphere => N {\x.and_(object(x),sphere(x))}
  donut => N {\x.and_(object(x),donut(x))}
  hose => N {\x.and_(object(x),hose(x))}
  cylinder => N {\x.and_(object(x),cylinder(x))}
  pyramid => N {\x.and_(object(x),pyramid(x))}

  the => N/N {\x.unique(x)}

  const => S {near}

  give => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,any))}
  send => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,far))}
  hand => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,near))}
  """, include_semantics=True)

from clevros.primitives import *

scene = {
    "objects": [
      frozendict({"female": True, "agent": True, "shape": "person"}),
      frozendict({"shape": "donut"}),
      frozendict({"shape": "cube"}),
      ]
    }
examples = [
    ("gorp the woman the cube", scene,
     ComposedAction(CausePossession(scene["objects"][0], {"shape": "cube"}), Transfer({"shape": "cube"}, scene["objects"][0], "far"))),
]

types = TypeSystem(["obj", "num", "ax", "dist", "boolean", "action"])

functions = [
  types.new_function("cmp_pos", ("ax", "obj", "obj", "num"), fn_cmp_pos),
  types.new_function("ltzero", ("num", "boolean"), fn_ltzero),
  types.new_function("and_", ("boolean", "boolean", "boolean"), fn_and),

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

  types.new_function("male", ("obj", "boolean"), lambda x: True), # TODO
  types.new_function("female", ("obj", "boolean"), lambda x: True), # TODO

  types.new_function("object", (types.ANY_TYPE, "boolean"), fn_object),
  types.new_function("agent", (types.ANY_TYPE, "boolean"), lambda x: True), # TODO

  types.new_function("move", ("obj", ("obj", "boolean"), "action"), lambda obj, dest: Move(a, b)),
  types.new_function("cause_possession", ("obj", "obj", "action"), lambda agent, obj: CausePossession(agent, obj)),
  types.new_function("transfer", ("obj", "obj", "dist", "action"), lambda obj, agent, dist: Transfer(obj, agent, dist)),

  types.new_function("do_", ("action", "action", "action"), lambda a1, a2: a1 + a2),
]

constants = [types.new_constant("any", "dist"),
             types.new_constant("far", "dist"),
             types.new_constant("near", "dist")]

ontology = Ontology(types, functions, constants, variable_weight=0.1)
compressor = Compressor(ontology, **EC_kwargs)

def compress_lexicon(lex):
  # Run EC compression on the entries of the induced lexicon. This may create
  # new inventions, updating both the `ontology` and the provided `lex`.
  lex, affected_entries = compressor.make_inventions(lex)

  for invention_name, tokens in affected_entries.items():
    affected_syntaxes = set(t.categ() for t in tokens)
    if len(affected_syntaxes) == 1:
      # Just one syntax is involved. Create a new derived category.
      derived_name = lex.add_derived_category(tokens)
      lex.propagate_derived_category(derived_name)

      print("Created and propagated derived category %s == %s -- %r" %
            (derived_name, lex._derived_categories[derived_name][0].base, tokens))

  return lex


#############

if __name__ == "__main__":
  # Run compression on the initial lexicon.
  lex = compress_lexicon(lex)


  for sentence, scene, answer in examples:
    sentence = sentence.split()

    model = Model(scene, ontology)
    parse_results = WeightedCCGChartParser(lex).parse(sentence)
    if not parse_results:
      print("ERROR: Parse failed for sentence '%s'" % " ".join(sentence))

      query_tokens = [word for word in sentence if not lex._entries.get(word, [])]
      print("\tNovel words: ", " ".join(query_tokens))
      query_token_syntaxes = get_candidate_categories(lex, query_tokens, sentence)
      print("\tCandidate categories:", query_token_syntaxes)

      # Augment the lexicon with all entries for novel words which yield the
      # correct answer to the sentence under some parse. Restrict the search by
      # the supported syntaxes for the novel words (`query_token_syntaxes`).
      lex = augment_lexicon_distant(lex, query_tokens, query_token_syntaxes,
                                    sentence, ontology, model, answer)

      lex = compress_lexicon(lex)

      parse_results = WeightedCCGChartParser(lex).parse(sentence)
      #print("parse_results:", parse_results)

    final_sem = parse_results[0].label()[0].semantics()

    sys.exit()

    print(" ".join(sentence), len(parse_results), final_sem)
    print("\t", model.evaluate(final_sem))

