"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

from nltk.ccg import chart
import numpy as np

from clevros.chart import WeightedCCGChartParser
from clevros.lexicon import Lexicon, augment_lexicon, \
    filter_lexicon_entry, augment_lexicon_scene
from clevros.perceptron import update_perceptron_batch
from clevros.rsa import infer_listener_rsa, update_weights_rsa


# Teeny subset of CLEVR dataset :)
scenes = [
  {'directions': {'above': [0.0, 0.0, 1.0],
                  'behind': [-0.754490315914154, 0.6563112735748291, 0.0],
                  'below': [-0.0, -0.0, -1.0],
                  'front': [0.754490315914154, -0.6563112735748291, -0.0],
                  'left': [-0.6563112735748291, -0.7544902563095093, 0.0],
                  'right': [0.6563112735748291, 0.7544902563095093, -0.0]},
 'image_filename': 'CLEVR_train_000002.png',
 'image_index': 2,
 'objects': [{'3d_coords': [1.9671114683151245,
                            1.1988708972930908,
                            0.3499999940395355],
              'color': 'gray',
              'material': 'metal',
              'pixel_coords': [350, 163, 10.398611068725586],
              'rotation': 2.1325644261237775,
              'shape': 'sphere',
              'size': 'small'},
             {'3d_coords': [2.1141371726989746,
                            -0.5752051472663879,
                            0.699999988079071],
              'color': 'yellow',
              'material': 'metal',
              'pixel_coords': [291, 180, 9.147775650024414],
              'rotation': 308.49217566676606,
              'shape': 'sphere',
              'size': 'large'},
             {'3d_coords': [-2.3854215145111084,
                            0.32431545853614807,
                            0.699999988079071],
              'color': 'blue',
              'material': 'rubber',
              'pixel_coords': [188, 94, 12.699371337890625],
              'rotation': 82.51702981683107,
              'shape': 'cube',
              'size': 'large'}],
 'relationships': {'behind': [[2], [0, 2], []],
                   'front': [[1], [], [0, 1]],
                   'left': [[1, 2], [2], []],
                   'right': [[], [0], [0, 1]]},
 'split': 'train'},
]


semantics = True

lex = Lexicon.fromstring(r"""
  :- NN, DET, ADJ

  DET :: NN/NN
  ADJ :: NN/NN

  the => DET {\x.unique(x)}
  sphere => NN {filter_shape(scene,sphere)}
  """, include_semantics=semantics)


sentence = "the cube".split()

# Now augment lexicon to account for new data.
lex_aug = augment_lexicon_scene(lex, sentence, scenes[0])
#lex_aug = filter_lexicon_entry(lex_aug, "blue", data_phase2[0][0], data_phase2[0][1])
lex_aug = update_weights_rsa(lex_aug, "cube")
print(lex_aug)

# Demo.
parser = WeightedCCGChartParser(lex_aug)
results = parser.parse(sentence)
chart.printCCGDerivation(results[0])
