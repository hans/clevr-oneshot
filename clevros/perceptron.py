"""
Structured perceptron algorithm for learning CCG weights.
"""

from collections import Counter
import logging
import numpy as np

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


def update_perceptron(lexicon, sentence, model, success_fn,
                      learning_rate=10, parser=None):
  if parser is None:
    parser = chart.WeightedCCGChartParser(lexicon,
                                          ruleset=chart.DefaultRuleSet)

  norm = 0.0
  weighted_results = parser.parse(sentence, return_aux=True)
  if not weighted_results:
    raise ValueError("No successful parses computed.")

  max_score, max_incorrect_score = -np.inf, -np.inf
  correct_results, incorrect_results = [], []

  for result, score, _ in weighted_results:
    if success_fn(result, model):
      if score > max_score:
        max_score = score
        correct_results = [(score, result)]
      elif score == max_score:
        correct_results.append((score, result))
    else:
      if score > max_incorrect_score:
        max_incorrect_score = score
        incorrect_results = [(score, result)]
      elif score == max_incorrect_score:
        incorrect_results.append((score, result))

  if not correct_results:
    raise ValueError("No parses derived are successful.")
  elif not incorrect_results:
    L.warning("No incorrect parses. Skipping update.")
    return weighted_results, 0.0

  # Sort results by descending parse score.
  correct_results = sorted(correct_results, key=lambda r: r[0], reverse=True)
  incorrect_results = sorted(incorrect_results, key=lambda r: r[0], reverse=True)

  # TODO margin?

  # Update to separate max-scoring parse from max-scoring correct parse if
  # necessary.
  positive_mass = 1 / len(correct_results)
  negative_mass = 1 / len(incorrect_results)

  token_deltas = Counter()
  observed_leaf_sequences = set()
  for results, delta in zip([correct_results, incorrect_results],
                             [positive_mass, -negative_mass]):
    for _, result in results:
      leaf_seq = tuple(leaf_token for _, leaf_token in result.pos())
      if leaf_seq not in observed_leaf_sequences:
        observed_leaf_sequences.add(leaf_seq)
        for leaf_token in leaf_seq:
          token_deltas[leaf_token] += delta

  for token, delta in token_deltas.items():
    delta *= learning_rate
    norm += delta ** 2

    L.info("Applying delta: %+.03f %s", delta, token)
    token._weight += delta

  return weighted_results, norm


def update_perceptron_distant(lexicon, sentence, model, answer,
                              **update_perceptron_kwargs):
  def success_fn(parse_result, model):
    root_token, _ = parse_result.label()

    try:
      pred_answer = model.evaluate(root_token.semantics())
      success = pred_answer == answer
    except (TypeError, AttributeError) as e:
      # Type inconsistency. TODO catch this in the iter_expression
      # stage, or typecheck before evaluating.
      success = False
    except AssertionError as e:
      # Precondition of semantics failed to pass.
      success = False

    return success

  L.debug("Desired answer: %s", answer)
  return update_perceptron(lexicon, sentence, model, success_fn,
                           **update_perceptron_kwargs)


def update_perceptron_2afc(lexicon, sentence, models,
                           **update_perceptron_kwargs):
  def success_fn(parse_result, models):
    model1, model2 = models
    root_token, _ = parse_result.label()
    sentence_semantics = root_token.semantics()

    try:
      model1_success = model1.evaluate(sentence_semantics) == True
    except:
      model1_success = False
    try:
      model2_success = model2.evaluate(sentence_semantics) == True
    except:
      model2_success = False

    return model1_success or model2_success

  return update_perceptron(lexicon, sentence, models, success_fn,
                           **update_perceptron_kwargs)
