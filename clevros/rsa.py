"""
RSA implementations operating over a CCG.
"""

from collections import defaultdict

import numpy as np


def infer_listener_rsa(lexicon, entry):
  """
  Infer the semantic form of an entry in a weighted lexicon via RSA
  inference.

  Args:
    lexicon: CCGLexicon
    entry: string word

  Returns:
    semantics: list of Expressions
    weights: corresponding inferred weights
  """

  # derive literal listener weights
  # p(s|u)
  ll_tokens = lexicon._entries[entry]
  ll_weights = np.array([token.weight() for token in ll_tokens])
  ll_weights /= ll_weights.sum()
  ll_weights = {token.semantics(): weight for token, weight in zip(ll_tokens, ll_weights)}

  # derive pragmatic speaker weights
  # p(u|s)
  speaker_weights = defaultdict(dict)
  for semantics, weight in ll_weights.items():
    # gather possible ways to express the meaning
    # TODO cache this reverse lookup?
    semantic_weights = {}

    for alt_word, alt_tokens in lexicon._entries.items():
      for alt_token in alt_tokens:
        if alt_token.semantics() == semantics:
          semantic_weights[alt_token._token] = alt_token.weight()

    # Normalize.
    total = sum(semantic_weights.values())
    speaker_weights[semantics] = {k: v / total
                                  for k, v in semantic_weights.items()}

  # derive pragmatic listener weights: transpose and renormalize
  pl_weights = {}
  for semantics, word_weights in speaker_weights.items():
    total = 0
    for word, weight in word_weights.items():
      if word == entry:
        pl_weights[semantics] = weight
      total += weight

    pl_weights = {k: v / total for k, v in pl_weights.items()}

  return pl_weights.keys(), pl_weights.values()
