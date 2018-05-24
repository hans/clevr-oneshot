"""
Functional tests for the Levin verb learning environment.
"""

import copy
import unittest

from nose.plugins.attrib import attr

from clevros.environments import vol
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

    # OK, now try to bootstrap with an example.
