"""
Functional tests for the Levin verb learning environment.
"""

import copy
import unittest

from nose.plugins.attrib import attr

from clevros.environments import levin
from clevros.word_learner import WordLearner


@attr(speed="slow")
class LevinTest(unittest.TestCase):

  def setUp(self):
    self.lexicon = levin.lexicon.clone()
    self.ontology = copy.deepcopy(levin.ontology)
    self.learner = WordLearner(self.lexicon, self.ontology,
                               compress=True, bootstrap=True)

  def test_compressor(self):
    # We should get five derived categories.
    self.learner.compress_lexicon()
    self.assertEquals(len(self.learner.lexicon._derived_categories), 5)
