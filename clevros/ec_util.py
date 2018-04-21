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
from ec import Grammar
from ec import Primitive

def ontology_to_grammar(ontology):

	#get a list of types for all prims
	tps = [len(inspect.getargspec(fn).args) for fn in ontology.function_names]

	#zip primitive names, types, and defs
	zipped_o = zip(ontology.function_names, tps, ontology.function_defs)

	#make into prim list
	primitives = [Primitive(name, tp, function) for name, tp, function in zipped_o ] 

	#zip into productions 
	productions = zip(ontology.function_weights, primitives)
	
	#return Grammar(logVariable, [(l, p.infer(), p) for l, p in productions])
	grammar = Grammar.fromProductions(productions, logVariable=ontology.logVariable)
	return grammar

def extract_frontiers_from_lexicon(lex):

	

	return frontiers

def grammar_to_ontology(grammar):

	return ontology

def frontiers_to_lexicon(frontiers):

	return lexicon 


