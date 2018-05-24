"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

from argparse import ArgumentParser
import inspect
import logging
import numbers
from pprint import pprint

from frozendict import frozendict
from nltk.ccg import chart
import numpy as np

from clevros.chart import WeightedCCGChartParser, DefaultRuleSet
from clevros.lexicon import Lexicon, Token, \
    augment_lexicon_distant, get_candidate_categories
from clevros.logic import Ontology, Function, TypeSystem
from clevros.model import Model
from clevros.perceptron import update_perceptron_distant
from clevros.compression import Compressor

import random
random.seed(4)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")
L = logging.getLogger(__name__)


#compression params:
EC_kwargs = {
  "topK": 1,
  "pseudoCounts": 1.0,
  "a": 1,
  "aic": -1.0,
  "structurePenalty": 0.001,
  "backend": "pypy",
  "CPUs": 1,
}


def compress_lexicon(lex, compressor):
  # Run EC compression on the entries of the induced lexicon. This may create
  # new inventions, updating both the `ontology` and the provided `lex`.
  lex, affected_entries = compressor.make_inventions(lex)

  for invention_name, tokens in affected_entries.items():
    if invention_name in lex._derived_categories_by_source:
      continue

    affected_syntaxes = set(t.categ() for t in tokens)
    if len(affected_syntaxes) == 1:
      # Just one syntax is involved. Create a new derived category.
      L.debug("Creating new derived category for tokens %r", tokens)

      derived_name = lex.add_derived_category(tokens, source_name=invention_name)
      lex.propagate_derived_category(derived_name)

      L.info("Created and propagated derived category %s == %s -- %r",
             derived_name, lex._derived_categories[derived_name][0].base, tokens)

  return lex


#############

def main(args, lex, ontology, examples):
  if not args.no_compress:
    # Run compression on the initial lexicon.
    compressor = Compressor(ontology)
    lex = compress_lexicon(lex, compressor)

  for sentence, scene, answer in examples:
    print("\n\n")

    sentence = sentence.split()

    model = Model(scene, ontology)

    try:
      weighted_results, _ = update_perceptron_distant(lex, sentence, model, answer)
    except ValueError:
      # No parse succeeded -- attempt lexical induction.
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))

      query_tokens = [word for word in sentence if not lex._entries.get(word, [])]
      L.info("Novel words: %s", " ".join(query_tokens))
      query_token_syntaxes = get_candidate_categories(lex, query_tokens, sentence)

      # Augment the lexicon with all entries for novel words which yield the
      # correct answer to the sentence under some parse. Restrict the search by
      # the supported syntaxes for the novel words (`query_token_syntaxes`).
      lex = augment_lexicon_distant(lex, query_tokens, query_token_syntaxes,
                                    sentence, ontology, model, answer,
                                    bootstrap=not args.no_bootstrap)

      if not args.no_compress:
        # Run compression on the augmented lexicon.
        lex = compress_lexicon(lex, compressor)

      # Attempt a new parameter update.
      weighted_results, _ = update_perceptron_distant(lex, sentence, model, answer)

    final_sem = weighted_results[0][0].label()[0].semantics()

    print(" ".join(sentence), len(weighted_results), final_sem)
    print("\t", model.evaluate(final_sem))


if __name__ == "__main__":
  p = ArgumentParser()

  # Model lesion options.
  p.add_argument("--no-compress", action="store_true")
  p.add_argument("--no-bootstrap", action="store_true")

  # from clevros.environments import voo
  # main(p.parse_args(), voo.lexicon, voo.ontology, voo.examples)

  # from clevros.environments import vol
  # main(p.parse_args(), vol.lexicon, vol.ontology, vol.examples)

  from clevros.environments import levin
  main(p.parse_args(), levin.lexicon, levin.ontology, levin.examples)
