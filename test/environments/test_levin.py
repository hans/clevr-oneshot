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

  def test_functional(self):
    # We should get six derived categories -- one for each verb class.
    self.learner.compress_lexicon()
    self.assertEquals(len(self.learner.lexicon._derived_categories), 6)

    # OK, now try to bootstrap with an example.
