"""
Functional tests for the Levin verb learning environment.
"""

import copy
import unittest

from nose.plugins.attrib import attr

from clevros.environments import vol
from clevros.model import Model
from clevros.word_learner import WordLearner


@attr(speed="slow")
class LevinTest(unittest.TestCase):

  def setUp(self):
    self.lexicon = vol.lexicon.clone()
    self.ontology = copy.deepcopy(vol.ontology)
    self.learner = WordLearner(self.lexicon, self.ontology,
                               compress=True, bootstrap=True)

  def test_functional(self):
    # We should get five derived categories.
    self.learner.compress_lexicon()
    self.assertEquals(len(self.learner.lexicon._derived_categories), 2)

    # Learn examples
    for sentence, scene, answer in vol.examples:
      sentence = sentence.split()
      model = Model(scene, self.ontology)

      weighted_results = self.learner.update_with_example(sentence, model, answer)

      final_sem = weighted_results[0][0].label()[0].semantics()

      print(" ".join(sentence), len(weighted_results), final_sem)
      print("\t", model.evaluate(final_sem))

