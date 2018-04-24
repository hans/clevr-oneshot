"""
Utilities for going back and forth between EC and CCG frameworks
"""

import inspect
import numbers
from pprint import pprint

from frozendict import frozendict
from nltk.ccg import chart
from nltk.ccg.api import PrimitiveCategory

import numpy as np

from clevros.chart import WeightedCCGChartParser
from clevros.lexicon import Lexicon, augment_lexicon, \
    filter_lexicon_entry, augment_lexicon_scene, augment_lexicon_distant, \
    get_candidate_categories, Token
from clevros.logic import Ontology, as_ec_sexpr, read_ec_sexpr 
from clevros.model import Model
from clevros.perceptron import update_perceptron_batch
from clevros.rsa import infer_listener_rsa, update_weights_rsa


#TODO:
#ec imports
from ec.grammar import Grammar
from ec.task import Task
from ec.frontier import Frontier, FrontierEntry
from ec.type import baseType, arrow
from ec.program import Primitive, Program

tS = baseType("S")

def convert_to_ec_type(arity):
	if arity == 0:
		tp = tS
	else:
		tp = arrow(*[tS for _ in range(arity + 1)])
	return tp



def ontology_to_grammar_initial(ontology):
	"""
	turns an ontology into a grammar. 
	This should only be done once per experiment. 
	Otherwise, it will mess with the fact that EC uses the knowledge 
	that some productions are prims and some are invented
	"""

	#get a list of types for all prims
	#we will change this when we have more sophisticated types
	tps = [convert_to_ec_type(len(inspect.getargspec(fn).args)) for fn in ontology.function_defs]

	#zip primitive names, types, and defs
	zipped_ont = zip(ontology.function_names, tps, ontology.function_defs)

	#make into prim list
	primitives = [Primitive(name, tp, function) for name, tp, function in zipped_ont ] 

	#zip into productions 
	productions = zip(ontology.function_weights, primitives)
	
	#return Grammar(logVariable, [(l, p.infer(), p) for l, p in productions])
	grammar = Grammar.fromProductions(productions, logVariable=ontology.variable_weight)
	return grammar


def grammar_to_ontology(grammar):
	#print(grammar.productions)

	#unzip productions into weights and primitives 
	weights_and_programs = list(zip(*grammar.productions))
	function_weights = weights_and_programs[0]
	programs = weights_and_programs[2]


	#names and defs
	function_names = [prim.show("error") for prim in programs]

	function_defs = []
	for prog in programs:
		if prog.isPrimitive:
			function_defs.append(prog.value)
		elif prog.isInvented:
			print("%s"%(prog.body.show(False)))
			function_defs.append(read_ec_sexpr("%s"%(prog.body.show(False)))[0])
		else: 
			print("not primitive or invented")
			assert False

	print("function_names",function_names)
	print("function_defs",function_defs)

	#function_names = remove_hashtags(function_names)

	ontology = Ontology(function_names, function_defs, function_weights, variable_weight=grammar.logVariable)
	return ontology

def get_category_arity(cat):
	#takes a category .categ() as input
	if isinstance(cat, PrimitiveCategory):
		return 0
	else:
		return 1 + get_category_arity(cat.arg()) \
			+ get_category_arity(cat.res())

def get_semantic_arity(cat):
	cat_ar = get_category_arity(cat)
	if cat_ar == 0:
		return 1
	else:
		return cat_ar




def extract_frontiers_from_lexicon(lex, g):
	frontiers = []
	for key in lex._entries:


		#for now, assume only one type per word in lexicon:
		for entry in lex._entries[key]:
			assert get_semantic_arity(entry.categ()) == get_semantic_arity(lex._entries[key][0].categ())

		#print("arity:", get_semantic_arity(lex._entries[key][0].categ()))

		request = convert_to_ec_type(get_semantic_arity(lex._entries[key][0].categ()))
		#print("request:")
		#print(request)

		task = Task(key, request, [])
		#print("key:", key)

		#the following line won't work because first input to FrontierEntry must be a Program
		#need function extract_s_exp

		#this will likely be changed
		def program(x):
			p = Program.parse(as_ec_sexpr(x))
			#print("program:")
			#print(str(p))
			return p

		#logLikelihood is 0.0 because we assume that it has parsed correctly already - may want to modify
		frontier_entry_list = [FrontierEntry(program(entry.semantics()), logPrior=g.logLikelihood(request, program(entry.semantics())), logLikelihood=0.0) for entry in lex._entries[key]]

		frontier = Frontier(frontier_entry_list, task)
		frontiers.append(frontier)

		#FrontierEntry(Program.parse(s["expression"]), logPrior=s["logprior"], logLikelihood=s["loglikelihood"])
		#lexicon._entries["block"][0].weight()
		#lexicon._entries["block"][0].semantics()

	return frontiers


def frontiers_to_lexicon(frontiers, old_lex):
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


	"""
	WARNING!!!
	The code below assumes that compression does not reorder the FrontierEntry's of a Frontier. 
	This might not be accurate.
	Code may break if there are multiple entries per frontier
	"""

	lex = old_lex.clone()

	for frontier in frontiers:
		word = frontier.task.name
		lex._entries[word] = []
		for frontier_entry, lex_entry in zip(frontier.entries, old_lex._entries[word]):
			#frontier_entry is a FrontierEntry
			#lex_entry is a Token
			#print("show", frontier_entry.program.show(False))
			semantics = read_ec_sexpr(frontier_entry.program.show(False))[0]
			#print("semantics", semantics)
			token = Token(word, lex_entry.categ(), semantics, lex_entry.weight())

			lex._entries[word].append(token)
	return lex

	"""
    Class representing a token.

    token => category {semantics}
    e.g. eat => S\\var[pl]/var {\\x y.eat(x,y)}

    * `token` (string) word
    * `categ` (string) syntactic type - .categ obj
    * `weight` (float) - 
    * `semantics` (Expression) - 
    """
	"""
    def __init__(self, token, categ, semantics=None, weight=1.0):
        self._token = token
        self._categ = categ
        self._weight = weight
        self._semantics = semantics
	"""

	#def __init__(self, start, primitives, families, entries):
     #   self._start = PrimitiveCategory(start)
      #  self._primitives = primitives
      #  self._families = families
       # self._entries = entries
       #deepcopy families, primitives and start
       #see augment_lex for appending new entries 




