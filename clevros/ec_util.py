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

def ontology_to_grammar_initial(ontology):
	"""
	turns an ontology into a grammar. 
	This should only be done once per experiment. 
	Otherwise, it will mess with the fact that EC uses the knowledge 
	that some productions are prims and some are invented
	"""

	#get a list of types for all prims
	#we will change this when we have more sophisticated types
	tps = [len(inspect.getargspec(fn).args) for fn in ontology.function_names]

	#zip primitive names, types, and defs
	zipped_ont = zip(ontology.function_names, tps, ontology.function_defs)

	#make into prim list
	primitives = [Primitive(name, tp, function) for name, tp, function in zipped_ont ] 

	#zip into productions 
	productions = zip(ontology.function_weights, primitives)
	
	#return Grammar(logVariable, [(l, p.infer(), p) for l, p in productions])
	grammar = Grammar.fromProductions(productions, logVariable=ontology.logVariable)
	return grammar


def grammar_to_ontology(grammar):

	#unzip productions into weights and primitives 
	weights_and_prims = zip(*grammar.productions)
	function_weights = weights_and_prims[0]
	primitives = weights_and_prims[1]

	#names and defs
	function_names = [prim.name for prim in primitives]
	function_defs = [prim.value for prim in primitives]

	#TODO
	#the following two lines will have to be fixed somehow
	ontology = Ontology(function_names, function_defs, function_weights)
	ontology.logVariable = grammar.logVariable 

	return ontology


def extract_frontiers_from_lexicon(lex):


	return frontiers


def frontiers_to_lexicon(frontiers):
	"""
	frontier has properties:
		frontier.entries, a list of frontierEntry's (presumably)
		frontier.task
	
	class FrontierEntry(object):
    def __init__(self, program, _=None, logPrior=None, logLikelihood=None, logPosterior=None):

	class Task(object):
    	def __init__(self, name, request, examples, features=None, cache=False):


   	from compressor we can see (https://github.com/ellisk42/ec/blob/480b51bb56f583ec5332608f054bf934db67cd66/fragmentGrammar.py#L396)
	need 

	"""



	return lexicon 


