"""
Functional tests for the Levin verb learning environment.
"""

import copy
import unittest

from nose.plugins.attrib import attr

from clevros.compression import Compressor
from clevros.environments import levin
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
    # We should get six derived categories -- one for each verb class.
    self.learner.compress_lexicon()
    self.assertEquals(len(self.learner.lexicon._derived_categories), 6)

    # OK, now try to bootstrap with an example.
