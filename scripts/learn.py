"""
Given an ambiguous lexicon and fully supervised training data, run a basic CCG
perceptron-style inference and update lexicon weights.
"""

from nltk.ccg import chart, lexicon

import numpy as np


semantics = True

lex = lexicon.fromstring(r"""
    :- NN, DET, ADJ

    DET :: NN/NN
    ADJ :: NN/NN

    the => DET {\x.unique(x)}

    blue => ADJ {\x.color(x,red)}
    blue => ADJ {\x.color(x,blue)}

    red => ADJ {\x.color(x,red)}
    red => ADJ {\x.color(x,blue)}

    ball => NN {ball}
    """, include_semantics=semantics)


data = [
    ("the blue ball".split(), "unique(color(ball,blue))"),
    ("the red ball".split(), "unique(color(ball,red))"),
]


def learn(lexicon, data):
    parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)

    learning_rate = 0.1
    for x, y in data:
        weighted_results = parser.parse(x, return_weights=True)

        # Very dumb perceptron learning
        for result, score in weighted_results:
            print("\n================= %s / %s / %f" %
                  (" ".join(x), result.label()[0].semantics(), score))
            chart.printCCGDerivation(result)

            root_token, _ = result.label()
            correct = str(root_token.semantics()) == y
            sign = 1 if correct else -1

            for _, leaf_token in result.pos():
                leaf_token._weight += sign * 1

        print()


for _ in range(2):
    learn(lex, data)
    print("\n\n")
