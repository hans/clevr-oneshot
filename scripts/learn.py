"""
Given an ambiguous lexicon and fully supervised training data, run a basic CCG
perceptron-style inference and update lexicon weights.
"""

from nltk.ccg import chart, lexicon

import numpy as np


class WeightedCCGLexicon(lexicon.CCGLexicon):
    """
    A CCG lexicon which maintains a decoupled efficient representation of the
    weights of its tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self._weights = np.zeros(len(self._entries))
        # for i, entry in enumerate(self._entries):
        #     self._weights[i] = entry.weight()
        self._weights = {entry: token.weight()
                         for entry, token in self._entries.items()}

        print(self._weights)


question = "the pen"
parse = "unique(utensil)"

semantics = True

lex = lexicon.fromstring(r"""
    :- NN, DET

    DET :: NN/NN

    the => DET {\x.unique(x)}

    pen => NN {utensil}
    pen => NN {area}
    """, include_semantics=semantics)

parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
results = parser.parse(question.split())

for result in results:
    chart.printCCGDerivation(result)
    token, _ = result.label()
    print(token.semantics())


def learn(lexicon, data):
    learning_rate = 0.5
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


learn(lex, [("the pen".split(), "unique(utensil)")])
print()
learn(lex, [("the pen".split(), "unique(utensil)")])
