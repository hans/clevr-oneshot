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
from clevros.lexicon import Lexicon, augment_lexicon, \
    filter_lexicon_entry, augment_lexicon_scene, augment_lexicon_distant, \
    get_candidate_categories, Token
from clevros.logic import Ontology
from clevros.model import Model
from clevros.perceptron import update_perceptron_batch
from clevros.rsa import infer_listener_rsa, update_weights_rsa
from clevros.ec_util import extract_frontiers_from_lexicon, \
    ontology_to_grammar_initial, grammar_to_ontology, frontiers_to_lexicon
from ec.fragmentGrammar import induceGrammar

import random
random.seed(4)

#compression params:
topK=10
pseudoCounts=1.0
arity=0
aic=1.0
structurePenalty=0.00001
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

  below => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,b,a))}
  below2 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,b,a))}
  behind =>  Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,b,a))}
  behind2 =>  Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,b,a))}
  right_of => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,b,a))}
  right_of2 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,b,a))}
  above => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  above2 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  left_of => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,a,b))}
  left_of2 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,a,b))}
  in_front_of => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,a,b))}
  in_front_of2 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,a,b))}

  """, include_semantics=semantics)

lex2 = Lexicon.fromstring(r"""
  :- S, Nd, N

  below3 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,b,a))}
  below4 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,b,a))}
  behind3 =>  Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,b,a))}
  behind4 =>  Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,b,a))}
  right_of3 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,b,a))}
  right_of4 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,b,a))}
  above3 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  above4 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_z,a,b))}
  left_of3 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,a,b))}
  left_of4 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_x,a,b))}
  in_front_of3 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,a,b))}
  in_front_of4 => Nd\Nd/Nd {\b.\a.ltzero(cmp_pos(ax_y,a,b))}


  """, include_semantics=True)

lex3 = Lexicon.fromstring(r"""
  :- S, Nd, N

  below4 => Nd\Nd/Nd {\b.\a.gt(pos_z(a),pos_z(b))}
  behind4 =>  Nd\Nd/Nd {\b.\a.gt(pos_y(b),pos_y(a))}

  right_of4 => Nd\Nd/Nd {\b.\a.gt(pos_x(a),pos_x(b))}
  above4 => Nd\Nd/Nd {\b.\a.gt(pos_z(b),pos_z(a))}
  baz4 => Nd\Nd/Nd {\b.\a.gt(pos_z(b),pos_z(a))}
  left_of4 => Nd\Nd/Nd {\b.\a.gt(pos_x(a),pos_x(b))}
  bar4 => Nd\Nd/Nd {\b.\a.gt(pos_x(a),pos_x(b))}
  in_front_of4 => Nd\Nd/Nd {\b.\a.gt(pos_y(a),pos_y(b))}
  foo4 => Nd\Nd/Nd {\b.\a.gt(pos_y(a),pos_y(b))}

  right_of5 => Nd\Nd/Nd {\b.\a.lt(pos_x(a),pos_x(b))}
  above5 => Nd\Nd/Nd {\b.\a.lt(pos_z(b),pos_z(a))}
  baz5 => Nd\Nd/Nd {\b.\a.lt(pos_z(b),pos_z(a))}
  left_of5 => Nd\Nd/Nd {\b.\a.lt(pos_x(a),pos_x(b))}
  bar5 => Nd\Nd/Nd {\b.\a.lt(pos_x(a),pos_x(b))}
  in_front_of5 => Nd\Nd/Nd {\b.\a.lt(pos_y(a),pos_y(b))}
  foo5 => Nd\Nd/Nd {\b.\a.lt(pos_y(a),pos_y(b))}

  """, include_semantics=True)



def fn_unique(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]

def fn_lt(x, y):
  assert isinstance(x, numbers.Number) and not isinstance(x, bool)
  assert isinstance(y, numbers.Number) and not isinstance(y, bool)
  return x < y

def fn_gt(x, y):
  assert isinstance(x, numbers.Number) and not isinstance(x, bool)
  assert isinstance(y, numbers.Number) and not isinstance(y, bool)
  return x > y


functions = {
  "cmp_pos": lambda ax, a, b: a["3d_coords"][ax()] - b["3d_coords"][ax()],
  "ltzero": lambda x: x < 0,

  "ax_x": lambda: 0,
  "ax_y": lambda: 1,
  "ax_z": lambda: 2,

  "unique": fn_unique,

  "cube": lambda x: x["shape"] == "cube",
  "sphere": lambda x: x["shape"] == "sphere",
}


ontology = Ontology(list(functions.keys()), list(functions.values()),
                    [ 0. for _ in range(len(functions))], variable_weight=0.1)

grammar = ontology_to_grammar_initial(ontology)

#############
invented_name_dict = None
for i, (sentence, scene, answer) in enumerate(examples):
  sentence = sentence.split()

  if i == 1:
    print("========Adding entries for 1")
    for entry, values in lex2._entries.items():
      lex._entries[entry] = values
  # elif i == 2:
  #   print("========Adding entries for 2")
  #   for entry, values in lex3._entries.items():
  #     lex._entries[entry] = values

  model = Model(scene, ontology)
  parse_results = WeightedCCGChartParser(lex).parse(sentence)
  if not False:
    """
    print("ERROR: Parse failed for sentence '%s'" % " ".join(sentence))

    query_tokens = [word for word in sentence if not lex._entries.get(word, [])]
    print("\tNovel words: ", " ".join(query_tokens))
    query_token_syntaxes = get_candidate_categories(lex, query_tokens, sentence)
    print("\tCandidate categories:")

    # Augment the lexicon with all entries for novel words which yield the
    # correct answer to the sentence under some parse. Restrict the search by
    # the supported syntaxes for the novel words (`query_token_syntaxes`).
    lex = augment_lexicon_distant(lex, query_tokens, query_token_syntaxes,
                                  sentence, ontology, model, answer)
    """
    # TODO(max) document
    frontiers = extract_frontiers_from_lexicon(lex, grammar, invented_name_dict=invented_name_dict)

    # EC compression phrase. Induce new functions using the present grammar.
    grammar, new_frontiers = induceGrammar(grammar, frontiers, topK=topK,
                                           pseudoCounts=pseudoCounts, a=arity,
                                           aic=aic, structurePenalty=structurePenalty,
                                           backend=compressor, CPUs=CPUs)

    # Convert result back to an ontology, switching to a naming scheme that
    # plays nice with our setup here.
    ontology, invented_name_dict = grammar_to_ontology(grammar)

    lex = frontiers_to_lexicon(new_frontiers, lex, invented_name_dict) #I think grammar not necessary

    # Recreate model with the new ontology.
    model = Model(scene, ontology)

    parse_results = WeightedCCGChartParser(lex).parse(sentence)
    #print("parse_results:", parse_results)

  final_sem = parse_results[0].label()[0].semantics()

  print(" ".join(sentence), len(parse_results), final_sem)
  print("\t", model.evaluate(final_sem))
