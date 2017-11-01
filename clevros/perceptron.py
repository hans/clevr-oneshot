"""
Structured perceptron algorithm for learning CCG weights.
"""

from clevros.chart import WeightedCCGChartParser


def update_perceptron_batch(lexicon, data, learning_rate=0.1, parser=None):
    """
    Execute a batch perceptron weight update with the given training data.

    Args:
        lexicon: CCGLexicon with weights
        data: List of `(x, y)` tuples, where `x` is a list of string
            tokens and `y` is an LF string.
        learning_rate:

    Returns:
        l2 norm of total weight updates
    """

    if parser is None:
        parser = WeightedCCGChartParser(lexicon)

    norm = 0.0
    for x, y in data:
        weighted_results = parser.parse(x, return_weights=True)
        for result, score in weighted_results:
            root_token, _ = result.label()
            correct = str(root_token.semantics()) == y
            sign = 1 if correct else -1

            for _, leaf_token in result.pos():
                delta = sign * learning_rate
                norm += delta ** 2

                leaf_token._weight += delta

    return norm
