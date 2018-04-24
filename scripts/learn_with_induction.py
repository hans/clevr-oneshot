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
                            -0.57520514,
                            0.699999988079071),
              'color': 'yellow',
              'material': 'metal',
              'pixel_coords': (291, 180, 9.147775650024414),
              'rotation': 308.49217566676606,
              'shape': 'sphere',
              'size': 'large'}),
             frozendict({'3d_coords': (-2.3854215145111084,
                            -0.57520514,
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
  ("is the sphere above the cube", scene, False),
  ("is the cube right_of the sphere", scene, False),
  ("is the sphere right_of the cube", scene, True),
]


#####################


semantics = True

lex = Lexicon.fromstring(r"""
  :- S, Nd, N

  cube => N {\x.cube(x)}
  sphere => N {\x.sphere(x)}

  is => S/Nd {\x.x}

  the => Nd/N {\x.unique(x)}

  left_of => Nd\Nd/Nd {\b.\a.lt(pos_x(a),pos_x(b))}
  above => Nd\Nd/Nd {\b.\a.gt(pos_y(a),pos_y(b))}
  """, include_semantics=semantics)



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
  "lt": fn_lt,
  "gt": fn_gt,
  "pos_x": lambda a: a["3d_coords"][0],
  "pos_y": lambda a: a["3d_coords"][1],
  "pos_z": lambda a: a["3d_coords"][2],
  "unique": fn_unique,

  "cube": lambda x: x["shape"] == "cube",
  "sphere": lambda x: x["shape"] == "sphere",
}


ontology = Ontology(functions.keys(), functions.values(), np.zeros(len(functions)), variable_weight=0.0)

grammar = ontology_to_grammar_initial(ontology)

#############

for sentence, scene, answer in examples:
  sentence = sentence.split()

  model = Model(scene, functions)
  parse_results = WeightedCCGChartParser(lex).parse(sentence)
  if not parse_results:
    print("ERROR: Parse failed for sentence '%s'" % " ".join(sentence))

    query_tokens = [word for word in sentence if not lex._entries.get(word, [])]
    print("\tNovel words: ", " ".join(query_tokens))
    query_token_syntaxes = get_candidate_categories(lex, query_tokens, sentence)
    print("\tCandidate categories:")
    pprint(query_token_syntaxes)

    lex = augment_lexicon_distant(lex, query_tokens, query_token_syntaxes,
                                  sentence, ontology, model, answer)

    #Max added
    frontiers = extract_frontiers_from_lexicon(lex, grammar)
    
    #EC compression phase 
    grammar, new_frontiers = induceGrammar(grammar, frontiers, topK=topK, 
                                           pseudoCounts=pseudoCounts, a=arity,
                                           aic=aic, structurePenalty=structurePenalty,
                                           backend=compressor, CPUs=CPUs)

    ontology = grammar_to_ontology(grammar)

    lex = frontiers_to_lexicon(new_frontiers, lex) #I think grammar not necessary

    parse_results = WeightedCCGChartParser(lex).parse(sentence)

  final_sem = parse_results[0].label()[0].semantics()

  print(" ".join(sentence), final_sem)
  print("\t", model.evaluate(final_sem))

