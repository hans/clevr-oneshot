"""
Structured perceptron algorithm for learning CCG weights.
"""

import logging

from clevros import chart

L = logging.getLogger(__name__)


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
    parser = chart.WeightedCCGChartParser(lexicon)

  norm = 0.0
  for x, y in data:
    weighted_results = parser.parse(x, return_aux=True)

    max_result, max_score, _ = weighted_results[0]
    correct_result, correct_score = None, None

    for result, score, _ in weighted_results:
      root_token, _ = result.label()
      if str(root_token.semantics()) == y:
        correct_result, correct_score = result, score
        break
    else:
      raise ValueError("no valid parse derived")

    if correct_score < max_score:
      for result, sign in zip([correct_result, max_result], [1, -1]):
        for _, leaf_token in result.pos():
          delta = sign * learning_rate
          norm += delta ** 2
          leaf_token._weight += delta

  return norm


def update_perceptron_distant(lexicon, sentence, model, answer,
                              learning_rate=0.1, parser=None):
  if parser is None:
    parser = chart.WeightedCCGChartParser(lexicon,
                                          ruleset=chart.DefaultRuleSet)

  norm = 0.0
  weighted_results = parser.parse(sentence, return_aux=True)
  print(len(weighted_results))
  if not weighted_results:
    raise ValueError("No successful parses computed.")

  correct_result, correct_score = None, None
  incorrect_results = []

  for result, score, _ in weighted_results:
    # Only track the max-scoring correct results and at least one incorrect
    # result which ranks near it.
    if incorrect_results and correct_result is not None:
      break

    root_token, _ = result.label()
    try:
      if model.evaluate(root_token.semantics()) == answer:
        correct_result, correct_score = result, score
      else:
        raise ValueError()
    except:
      incorrect_results.append(result)
  else:
    if correct_result is None:
      raise ValueError("No parses derived have the correct answer.")
    elif not incorrect_results:
      L.warning("No incorrect parses. Skipping update.")

  # Update to separate max-scoring parse from max-scoring correct parse if
  # necessary.
  negative_mass = 1 / len(incorrect_results)
  for results, delta in zip([[correct_result], incorrect_results],
                             [learning_rate, -negative_mass * learning_rate]):
    for result in results:
      for _, leaf_token in result.pos():
        norm += delta ** 2
        leaf_token._weight += delta

  return weighted_results, norm
