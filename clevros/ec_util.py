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

from collections import OrderedDict

tS = baseType("S")

tR = baseType("R")

def convert_to_ec_type_test(fn, name):

	if "pos_" in name:
		tp = arrow(tS,tR)
	elif name == "gt" or name == "lt":
		tp = arrow(tR, tR, tS)
	else:
		arity = len(inspect.getargspec(fn).args)

		if arity == 0:
			tp = tS
		else:
			tp = arrow(*[tS for _ in range(arity + 1)])
	#print("type:", tp)
	return tp



def convert_to_ec_type_vanilla(arity):

	if arity == 0:
		tp = tS
	else:
		tp = arrow(*[tS for _ in range(arity + 1)])
	return tp



def ontology_to_grammar_initial(ontology):
	"""
	turns an ontology into a grammar.
	This should only be done once per experiment.
	Otherwise, it will mess with the EC classes of invented vs primitives
	"""

	#get a list of types for all prims
	#we will change this when we have more sophisticated types

	tps = [convert_to_ec_type_test(fn, name) for fn, name in zip(ontology.function_defs, ontology.function_names)]

	#tps = [convert_to_ec_type_vanilla(len(inspect.getargspec(fn).args)) for fn in ontology.function_defs]

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

	#create ref_dict 
	prim_names = [prog.show("error") for prog in programs if prog.isPrimitive]

	prim_weights = [weight for weight,_,program in grammar.productions if program.isPrimitive]

	prim_defs = [prog.value for prog in programs if prog.isPrimitive]



	inv_weights = [weight for weight,_,program in grammar.productions if program.isInvented]

	inv_originals = [prog.show("error") for prog in programs if prog.isInvented]

	assert len(inv_weights) == len(inv_originals)

	inv_names = ["invented_" + str(i) for i in range(len(inv_originals))]

	#no hashtags 
	inv_defs = ["%s"%(prog.body.show(False)) for prog in programs if prog.isInvented]

	defs = OrderedDict(zip(inv_names,inv_defs))
	#originals = OrderedDict(zip(inv_names,inv_originals))
	originals = OrderedDict(sorted(zip(inv_names,inv_originals), key=lambda x: x[1].count("("), reverse=True))

	while any([("#" in defs[n]) for n in defs]):
		min_depth = min([inv_def.count("(") for inv_def in inv_defs])
		for name in inv_names:
			if defs[name].count("(") == min_depth:
				for n in inv_names:
					defs[n] = defs[n].replace(originals[name], name)

	inv_defs = [read_ec_sexpr(defs[name]) for name in defs]
	#print("inv_defs:\n",inv_defs)

	#names and defs
	"""
	function_defs = []
	#THIS IS WRONG
	#TODO
	#primitves output lambda functions, inventeds output lambdaexpressions. BAD.
	#ontology takes python lambda functions as function_defs
	for prog in programs:
		if prog.isPrimitive:
			function_defs.append(prog.value)

		elif prog.isInvented:
			print("%s"%(prog.body.show(False)))
			#want a way to read_ec_sexpr into python lambda function
			function_defs.append(read_ec_sexpr("%s"%(prog.body.show(False))))
		else:
			print("not primitive or invented")
			assert False

	print("function_names",function_names)
	print("function_defs",function_defs)
	"""

	function_names = prim_names + inv_names
	function_defs = prim_defs + inv_defs
	function_weights = prim_weights + inv_weights

	print("def:\n",function_defs)
	print("function_names:", function_names)
	#function_names = remove_hashtags(function_names)

	ontology = Ontology(function_names, function_defs, function_weights, variable_weight=grammar.logVariable)

	return ontology, originals #invented_name_dict



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




def extract_frontiers_from_lexicon(lex, g, invented_name_dict=None):
	frontiers = []
	for key in lex._entries:


		#for now, assume only one type per word in lexicon:
		for entry in lex._entries[key]:
			assert get_semantic_arity(entry.categ()) == get_semantic_arity(lex._entries[key][0].categ())
		#print("arity:", get_semantic_arity(lex._entries[key][0].categ()))

		request = convert_to_ec_type_vanilla(get_semantic_arity(lex._entries[key][0].categ()))
		#print("request:")
		#print(request)
		#print(lex._entries[key][0].categ())

		task = Task(key, request, [])
		#print("key:", key)
		#the following line won't work because first input to FrontierEntry must be a Program
		#need function extract_s_exp

		def program(x):
			x = as_ec_sexpr(x)
			if invented_name_dict:
				for name in invented_name_dict:
					x = x.replace(name, invented_name_dict[name])

			p = Program.parse(x)
			#print("program:")
			#print(str(p))
			return p

		#logLikelihood is 0.0 because we assume that it has parsed correctly already - may want to modify
		frontier_entry_list = [FrontierEntry(program(entry.semantics()), logPrior=g.logLikelihood(request, program(entry.semantics())), logLikelihood=0.0) for entry in lex._entries[key]]

		frontier = Frontier(frontier_entry_list, task)
		frontiers.append(frontier)

	return frontiers


def frontiers_to_lexicon(frontiers, old_lex, invented_name_dict):
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


			raw_program_str = frontier_entry.program.show(False)
			#sorting process
			for name in invented_name_dict:
				raw_program_str = raw_program_str.replace(invented_name_dict[name], name)



			semantics = read_ec_sexpr(raw_program_str)
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




