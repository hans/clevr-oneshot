"""
RSA implementations operating over a CCG.
"""

from collections import defaultdict
import copy

from nltk.ccg.lexicon import Token
import numpy as np


def infer_listener_rsa(lexicon, entry):
  """
  Infer the semantic form of an entry in a weighted lexicon via RSA
  inference.

  Args:
    lexicon: CCGLexicon
    entry: string word

  Returns:
    tokens: list of tokens with newly inferred weights
  """

  # derive literal listener weights
  # p(s|u)
  ll_tokens = lexicon._entries[entry]
  ll_weights = np.array([token.weight() for token in ll_tokens])
  ll_weights /= ll_weights.sum()
  ll_weights = dict(zip(ll_tokens, ll_weights))

  # derive pragmatic speaker weights
  # p(u|s)
  speaker_weights = defaultdict(dict)
  for token, weight in ll_weights.items():
    # gather possible ways to express the meaning
    # TODO cache this reverse lookup?
    semantic_weights = {}

    for alt_word, alt_tokens in lexicon._entries.items():
      for alt_token in alt_tokens:
        if alt_token.categ() == token.categ() \
            and alt_token.semantics() == token.semantics():
          semantic_weights[alt_token._token] = alt_token.weight()

    # Normalize.
    total = sum(semantic_weights.values())
    speaker_weights[token] = {k: v / total
                              for k, v in semantic_weights.items()}

  # derive pragmatic listener weights: transpose and renormalize
  pl_weights = {}
  for token, word_weights in speaker_weights.items():
    total = 0
    for word, weight in word_weights.items():
      if word == entry:
        pl_weights[token] = weight
      total += weight

  pl_weights = {k: v / total for k, v in pl_weights.items()}

  # Create a list of reweighted tokens
  new_tokens = [Token(token=entry, categ=t.categ(),
                      semantics=t.semantics(), weight=weight)
                for t, weight in pl_weights.items()]
  return new_tokens


def update_weights_rsa(lexicon, entry):
  """
  Update the weights of potential mappings of a lexicon entry by
  evaluating meanings under RSA.

  Returns a lexicon copy.

  Args:
    lexicon:
    entry:
  Returns:
    new_lexicon:
  """
  new_lex = copy.deepcopy(lexicon)
  new_lex._entries[entry] = infer_listener_rsa(lexicon, entry)
  return new_lex
