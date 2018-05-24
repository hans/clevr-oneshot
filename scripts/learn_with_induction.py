"""
Given an incomplete lexicon and fully supervised training data, run a
basic CCG perceptron-style inference and update lexicon weights.
"""

from argparse import ArgumentParser
import logging

from clevros.model import Model
from clevros.word_learner import WordLearner

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


#############

def main(args, lexicon, ontology, examples):
  compressor = Compressor(ontology, **EC_kwargs) if not args.no_compress else None
  learner = WordLearner(lexicon, compressor, bootstrap=not args.no_bootstrap)

  # No-op if `--no-compress`
  learner.compress_lexicon()

  for sentence, scene, answer in examples:
    print("\n\n")

    sentence = sentence.split()
    model = Model(scene, ontology)

    weighted_results = learner.update_with_example(sentence, model, answer)

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
