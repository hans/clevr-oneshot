"""
Utilities for going back and forth between EC and CCG frameworks
"""

import inspect
import numbers
from pprint import pprint

from frozendict import frozendict
from nltk.ccg import chart
import numpy as np

from clevros.chart import WeightedCCGChartParser
from clevros.lexicon import Lexicon, augment_lexicon, \
    filter_lexicon_entry, augment_lexicon_scene, augment_lexicon_distant, \
    get_candidate_categories
from clevros.logic import Ontology
from clevros.model import Model
from clevros.perceptron import update_perceptron_batch
from clevros.rsa import infer_listener_rsa, update_weights_rsa

#TODO:
#ec imports
from EC import Grammar
from EC import Primitive

def ontology_to_grammar(ontology):
	#TODO

	#Primitive(name, type, function)
	primitves = [Primitive() for ... in ontology ] 
	#will do uniform grammar for now, may need to change at some point 
	grammar = Grammar.uniform(primitives)
	return grammar

def extract_frontiers_from_lexicon(lex):

	return frontiers

def ontology_to_grammar(ontology):

	return grammar

def grammar_to_ontology(grammar):

	return ontology

def frontiers_to_lexicon(frontiers):

	return lexicon


