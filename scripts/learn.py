"""
Given an ambiguous lexicon and fully supervised training data, run a basic CCG
perceptron-style inference and update lexicon weights.
"""

from nltk.ccg import chart, lexicon

import numpy as np


semantics = True

lex = lexicon.fromstring(r"""
    :- NN, DET

    DET :: NN/NN

    the => DET {\x.unique(x)}

    blue => NN {red}
    blue => NN {blue}

    red => NN {red}
    red => NN {blue}
    """, include_semantics=semantics)


data = [
    ("the blue".split(), "unique(blue)"),
    ("the red".split(), "unique(red)"),
]


def learn(lexicon, data):
    parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)

    learning_rate = 0.1
    for x, y in data:
        weighted_results = parser.parse(x, return_weights=True)

        # Very dumb: upweight correct parse weights; downweight others
        for result, score in weighted_results:
            print("================= %f" % score)
            chart.printCCGDerivation(result)

            root_token, _ = result.label()
            correct = str(root_token.semantics()) == y
            sign = 1 if correct else -1

            for _, leaf_token in result.pos():
                leaf_token._weight += sign * 1


for _ in range(3):
    learn(lex, data)
