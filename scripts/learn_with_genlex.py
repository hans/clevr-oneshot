"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

from nltk.ccg import chart, lexicon
import numpy as np

from clevros.chart import WeightedCCGChartParser
from clevros.lexicon import augment_lexicon, filter_lexicon_entry
from clevros.perceptron import update_perceptron_batch
from clevros.rsa import infer_listener_rsa, update_weights_rsa


semantics = True

lex = lexicon.fromstring(r"""
  :- NN, DET, ADJ

  DET :: NN/NN
  ADJ :: NN/NN

  the => DET {\x.unique(x)}
  red => ADJ {\x.color(x,red)}
  ball => NN {ball}
  """, include_semantics=semantics)


data_phase1 = [
  ("the red ball".split(), "unique(color(ball,red))"),
]

data_phase2 = [
  ("the blue ball".split(), "unique(color(ball,blue))"),
]

# Strengthen weights for phase 1.
update_perceptron_batch(lex, data_phase1)

# Now augment lexicon to account for new data.
lex_aug = augment_lexicon(lex, data_phase2[0][0], data_phase2[0][1])
#lex_aug = filter_lexicon_entry(lex_aug, "blue", data_phase2[0][0], data_phase2[0][1])
lex_aug = update_weights_rsa(lex_aug, "blue")
print(lex_aug)

# Strengthen weights for phase 2.
update_perceptron_batch(lex_aug, data_phase2)

# Demo.
parser = WeightedCCGChartParser(lex_aug)
results = parser.parse(data_phase2[0][0])
chart.printCCGDerivation(results[0])
