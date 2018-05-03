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

  productions = [(fn.weight, Primitive(fn.name, convert_to_ec_type_vanilla(fn.arity), fn.defn))
                 for fn in ontology.functions]

  #return Grammar(logVariable, [(l, p.infer(), p) for l, p in productions])
  grammar = Grammar.fromProductions(productions, logVariable=ontology.variable_weight)
  return grammar


def grammar_to_ontology(grammar, ontology):
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

  # String-replace hashtag-style invention names from EC with more compact
  # names that are clevros-friendly.
  while any([("#" in defs[n]) for n in defs]):
    min_depth = min([inv_def.count("(") for inv_def in inv_defs])
    for name in inv_names:
      if defs[name].count("(") == min_depth:
        for n in inv_names:
          defs[n] = defs[n].replace(originals[name], name)

  # TODO refactor / simplify w.r.t. above ..
  # Convert invention definitions to native representation.
  ret_invs = []
  for inv_name, inv_defn, inv_weight in zip(inv_names, defs.values(), inv_weights):
    # First read an untyped version.
    inv_defn, bound_vars = read_ec_sexpr(inv_defn)

    # Run type inference on the bound variables.
    bound_signatures = {
      bound_var.name: ontology.infer_type(inv_defn, bound_var.name)
      for bound_var in bound_vars.values()
    }
    ontology.typecheck(inv_defn, bound_signatures)

    ret_invs.append(ontology.types.new_function(
      inv_name, inv_defn.type, inv_defn, weight=inv_weight))

  # TODO double-check that we don't end up with any ANY_TYPE

  ontology.add_functions(ret_invs)
  ontology.variable_weight = grammar.logVariable

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


    # TODO assumes that all lexical entries for a word have the same
    # syntactic arity.
    for entry in lex._entries[key]:
      assert get_semantic_arity(entry.categ()) == get_semantic_arity(lex._entries[key][0].categ())

    request = convert_to_ec_type_vanilla(get_semantic_arity(lex._entries[key][0].categ()))

    task = Task(key, request, [])
    #the following line won't work because first input to FrontierEntry must be a Program
    #need function extract_s_exp

    def program(x):
      x = as_ec_sexpr(x)
      if invented_name_dict:
        for name in invented_name_dict:
          x = x.replace(name, invented_name_dict[name])

      p = Program.parse(x)
      return p

    #logLikelihood is 0.0 because we assume that it has parsed correctly already - may want to modify
    frontier_entry_list = [FrontierEntry(program(entry.semantics()), logPrior=g.logLikelihood(request, program(entry.semantics())), logLikelihood=0.0) for entry in lex._entries[key]]

    frontier = Frontier(frontier_entry_list, task)
    frontiers.append(frontier)

  return frontiers


def frontiers_to_lexicon(frontiers, old_lex, invented_name_dict):
  """
  Convert old EC frontiers to a new `Lexicon` instance.

  Args:
    frontiers:
    old_lex: Prior `Lexicon` instance
    invented_name_dict: ordered dict returned from `grammar_to_ontology`
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
      raw_program_str = frontier_entry.program.show(False)

      # NB: it is important that `invented_name_dict` be ordered by
      # decreasing depth of the invented expressions.
      #
      # This way we ensure we attempt replacing with the largest
      # inventions first. Otherwise we might first replace a
      # sub-invention's expression and thereby kill our chance of later
      # replacing the containing invention's expression.
      for name in invented_name_dict:
        raw_program_str = raw_program_str.replace(invented_name_dict[name], name)

      semantics, _ = read_ec_sexpr(raw_program_str)
      #print("semantics", semantics)
      token = Token(word, lex_entry.categ(), semantics, lex_entry.weight())

      lex._entries[word].append(token)
  return lex
