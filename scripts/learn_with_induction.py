"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

from nltk.ccg import chart
import numpy as np

from clevros.chart import WeightedCCGChartParser
from clevros.lexicon import Lexicon, augment_lexicon, \
    filter_lexicon_entry, augment_lexicon_scene
from clevros.model import Model
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
 'objects': [
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
                            -0.57520514,
                            0.699999988079071],
              'color': 'blue',
              'material': 'rubber',
              'pixel_coords': [188, 94, 12.699371337890625],
              'rotation': 82.51702981683107,
              'shape': 'cube',
              'size': 'large'}],
 'split': 'train'},
]


semantics = True

lex = Lexicon.fromstring(r"""
  :- Nd, N

  cube => N {cube}
  sphere => N {sphere}

  the => Nd/N {\x.unique(x)}

  left_of => Nd\Nd/Nd {\b.\a.lt(pos_x(a),pos_x(b))}
  above => Nd\Nd/Nd {\b.\a.gt(pos_y(a),pos_y(b))}
  """, include_semantics=semantics)


sentence = "the cube left_of the sphere".split()

results = WeightedCCGChartParser(lex).parse(sentence)
chart.printCCGDerivation(results[0])

final_sem = results[0].label()[0].semantics()
print(final_sem)

def fn_unique(xs):
  assert len(xs) == 1
  return xs[0]

valuation_fns = {
  "cube": lambda scene: list(filter(lambda x: x["shape"] == "cube", scene["objects"])),
  "sphere": lambda scene: list(filter(lambda x: x["shape"] == "sphere", scene["objects"])),
}

functions = {
  "lt": lambda x, y: x < y,
  "gt": lambda x, y: x > y,
  "pos_x": lambda a: a["3d_coords"][0],
  "pos_y": lambda a: a["3d_coords"][1],
  "pos_z": lambda a: a["3d_coords"][2],
  "unique": fn_unique,
}

model = Model(valuation_fns, functions)
print(model.evaluate(final_sem, scenes[0]))

sys.exit(0)

# Now augment lexicon to account for new data.
lex_aug = augment_lexicon_scene(lex, sentence, scenes[0])
#lex_aug = filter_lexicon_entry(lex_aug, "blue", data_phase2[0][0], data_phase2[0][1])
lex_aug = update_weights_rsa(lex_aug, "cube")
print(lex_aug)

# Demo.
parser = WeightedCCGChartParser(lex_aug)
results = parser.parse(sentence)
chart.printCCGDerivation(results[0])
