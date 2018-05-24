"""
Functional tests for the Levin verb learning environment.
"""

import copy
import unittest

from nose.plugins.attrib import attr

from clevros.compression import Compressor
from clevros.environments import voo
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
    self.lexicon = voo.lexicon.clone()
    self.learner = WordLearner(self.lexicon, Compressor(self.lexicon.ontology, **EC_kwargs),
                               bootstrap=True)

  def test_functional(self):
    # We should get five derived categories.
    self.learner.compress_lexicon()
    self.assertEquals(len(self.learner.lexicon._derived_categories), 2)

    for sentence, scene, answer in voo.examples:
      sentence = sentence.split()
      model = Model(scene, self.learner.ontology)

      weighted_results = self.learner.update_with_example(sentence, model, answer)

      final_sem = weighted_results[0][0].label()[0].semantics()

      print(" ".join(sentence), len(weighted_results), final_sem)
      print("\t", model.evaluate(final_sem))
