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
from clevros.ec_util import extract_frontiers_from_lexicon, \
    ontology_to_grammar_initial, grammar_to_ontology, frontiers_to_lexicon
from ec.fragmentGrammar import induceGrammar

import random
random.seed(4)

#compression params:
topK=1
pseudoCounts=1.0
arity=0
aic=1.0
structurePenalty=0.001
compressor="rust" #"pypy"
CPUs=1



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
             frozendict(
               {'3d_coords': (2.1141371726989746,
                            1.0,
                            2),
              'color': 'yellow',
              'material': 'metal',
              'pixel_coords': (291, 180, 9.147775650024414),
              'rotation': 308.49217566676606,
              'shape': 'sphere',
              'size': 'large'}),
             frozendict({'3d_coords': (-2.3854215145111084,
                            0.0,
                            0.699999988079071),
              'color': 'blue',
              'material': 'rubber',
              'pixel_coords': (188, 94, 12.699371337890625),
              'rotation': 82.51702981683107,
              'shape': 'cube',
              'size': 'large'})],
 'split': 'train'}


examples = [
  ("is the cube left_of the sphere", scene, True),
  ("is the sphere left_of the cube", scene, False),
  ("is the cube above the sphere", scene, False),
  ("is the sphere above the cube", scene, True),
  ("is the cube right_of the sphere", scene, False),
  ("is the sphere right_of the cube", scene, True),
  ("is the cube below the sphere", scene, True),
  ("is the sphere below the cube", scene, False),
  ("is the sphere behind the cube", scene, True),
  ("is the cube behind the sphere", scene, False),
]


#####################


semantics = True

lex = Lexicon.fromstring(r"""
  :- S, Nd, N

  cube => N {\x.cube(x)}
  sphere => N {\x.sphere(x)}

  is => S/Nd {\x.x}

  the => Nd/N {\x.unique(x)}

  below => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  behind =>  Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,b,a))}

  above => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  left_of => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,a,b))}
  in_front_of => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,a,b))}
  """, include_semantics=semantics)


def fn_unique(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]

types = TypeSystem(["obj", "num", "ax", "boolean"])

functions = [
  types.new_function("cmp_pos", ("ax", "obj", "obj", "num"),
                     lambda ax, a, b: a["3d_coords"][ax()] - b["3d_coords"][ax()]),
  types.new_function("ltzero", ("num", "boolean"), lambda x: x < 0),

  types.new_function("ax_x", ("ax",), lambda: 0),
  types.new_function("ax_y", ("ax",), lambda: 1),
  types.new_function("ax_z", ("ax",), lambda: 2),

  types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),

  types.new_function("cube", ("obj", "boolean"), lambda x: x["shape"] == "cube"),
  types.new_function("sphere", ("obj", "boolean"), lambda x: x["shape"] == "sphere"),
]


ontology = Ontology(types, functions, variable_weight=0.1)

grammar = ontology_to_grammar_initial(ontology)

#############
invented_name_dict = None
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

    # TODO(max) document
    frontiers = extract_frontiers_from_lexicon(lex, grammar, invented_name_dict=invented_name_dict)

    # EC compression phrase. Induce new functions using the present grammar.
    grammar, new_frontiers = induceGrammar(grammar, frontiers, topK=topK,
                                           pseudoCounts=pseudoCounts, a=arity,
                                           aic=aic, structurePenalty=structurePenalty,
                                           backend=compressor, CPUs=CPUs)

    # Convert result back to an ontology, switching to a naming scheme that
    # plays nice with our setup here.
    ontology, invented_name_dict = grammar_to_ontology(grammar, ontology)

    lex = frontiers_to_lexicon(new_frontiers, lex, invented_name_dict) #I think grammar not necessary

    # Recreate model with the new ontology.
    model = Model(scene, ontology)

    parse_results = WeightedCCGChartParser(lex).parse(sentence)
    #print("parse_results:", parse_results)

  final_sem = parse_results[0].label()[0].semantics()

  print(" ".join(sentence), len(parse_results), final_sem)
  print("\t", model.evaluate(final_sem))

