"""
Functional tests for the Levin verb learning environment.
"""

import copy
import unittest

from nose.plugins.attrib import attr

from clevros.compression import Compressor
from clevros.environments import levin
from clevros.model import Model
from clevros.word_learner import WordLearner

EC_kwargs = {
  "topK": 1,
  "pseudoCounts": 1.0,
  "a": 1,
  "aic": -1.0,
  "structurePenalty": 0.001,
  "backend": "pypy",
  "CPUs": 1,
}

@attr(speed="slow")
class LevinTest(unittest.TestCase):

  def setUp(self):
    self.lexicon = levin.lexicon.clone()
    self.learner = WordLearner(self.lexicon, Compressor(self.lexicon.ontology, **EC_kwargs),
                               bootstrap=True)

  def test_functional(self):
    expected_derived = {
      # Levin verb classes
      "9.1": {"set", "put"},
      "9.2": {"lay", "hang"},
      "9.4": {"drop", "hoist"},
      "9.5": {"pour", "spill"},
      "9.7": {"spray", "load"},
      "9.8": {"fill", "stuff"},

      # PPs
      "contact": {"on", "onto"},
    }

    #######
    self.learner.compress_lexicon()
    #######

    self.assertEquals(set(frozenset(token._token for token in tokens)
                          for _, tokens in self.learner.lexicon._derived_categories.values()),
                      set(frozenset(xs) for xs in expected_derived.values()))

    # OK, now try to bootstrap with an example.
    for sentence, scene, answer in levin.examples:
      sentence = sentence.split()
      model = Model(scene, self.lexicon.ontology)

      weighted_results = self.learner.update_with_example(sentence, model, answer)

      final_sem = weighted_results[0][0].label()[0].semantics()

      print(" ".join(sentence), len(weighted_results), final_sem)
      print("\t", model.evaluate(final_sem))

