"""
Utilities for going back and forth between EC and CCG frameworks
"""

from collections import OrderedDict, defaultdict, namedtuple
import inspect

from nltk.ccg.api import PrimitiveCategory

from clevros.lexicon import Token, DerivedCategory
from clevros.logic import as_ec_sexpr, read_ec_sexpr


#TODO:
#ec imports
from ec.fragmentGrammar import induceGrammar
from ec.grammar import Grammar
from ec.task import Task
from ec.frontier import Frontier, FrontierEntry
from ec.type import baseType, arrow
from ec.program import Primitive, Program


tS = baseType("S")

tR = baseType("R")


Invention = namedtuple("Invention", ["name", "original_name", "weight", "defn"])


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
  # Prepare to add inventions from grammar as new ontology functions.
  inventions = [(prog.show("error"), weight, str(prog.body.show(False)))
                for weight, _, prog in grammar.productions if prog.isInvented]
  inventions = {"invented_%i" % i: Invention("invented_%i" % i, original, weight, program)
                for i, (original, weight, program) in enumerate(inventions)}

  new_defns = {name: inv.defn for name, inv in inventions.items()}

  # String-replace hashtag-style invention names from EC with more compact
  # names that are clevros-friendly.
  #
  # The string-replace happens bottom-up in order to support nested inventions.
  # Inventions which are subexpressions of other inventions get substituted
  # first.
  min_depth = -1
  # Track the depth ordering we end up using here -- will be useful in the near
  # future for type inference.
  depth_ordering = []
  while any([("#" in new_defn) for new_defn in new_defns.values()]):
    min_depth += 1
    for replace_source in inventions.values():
      if replace_source.defn.count("(") == min_depth:
        # This is one of the smallest remaining inventions --
        # search-and-replace in any other inventions which might contain it.
        for replace_target in inventions.values():
          new_defns[replace_target.name] = new_defns[replace_target.name] \
              .replace(replace_source.original_name, replace_source.name)

        depth_ordering.append(replace_source.name)
  # Highest-level inventions need to also be added to the depth ordering.
  depth_ordering.extend(set(inventions.keys()) - set(depth_ordering))

  # Convert invention definitions to native representation. Requires type
  # inference. Use the same increasing-depth ordering as before so that larger
  # inventions can do type inference knowing about the function types of
  # inventions they contain.
  invention_types = {}
  ret_invs = []
  for inv_name in depth_ordering:
    invention = inventions[inv_name]

    # First read an untyped version.
    inv_defn, bound_vars = read_ec_sexpr(new_defns[inv_name])

    # Run type inference on the bound variables.
    bound_signatures = {
      bound_var.name: ontology.infer_type(inv_defn, bound_var.name,
                                          extra_types=invention_types)
      for bound_var in bound_vars.values()
    }
    ontology.typecheck(inv_defn, bound_signatures)

    invention_types[inv_name] = inv_defn.type

    ret_invs.append(ontology.types.new_function(
      invention.name, inv_defn.type, inv_defn, weight=invention.weight))

  # TODO double-check that we don't end up with any ANY_TYPE

  ontology.add_functions(ret_invs)
  ontology.variable_weight = grammar.logVariable

  # NB: Ordered from maximum to minimum depth
  invented_name_dict = OrderedDict([(name, inventions[name].original_name)
                                    for name in depth_ordering[::-1]])

  return ontology, invented_name_dict



def get_category_arity(cat):
  #takes a category .categ() as input
  if isinstance(cat, DerivedCategory):
    return get_category_arity(cat.base)
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

    # DEBUG
    for entry in lex._entries[key]:
      print("%60s %20s %60s" % (entry, request, program(entry.semantics())))

    #logLikelihood is 0.0 because we assume that it has parsed correctly already - may want to modify
    frontier_entry_list = [FrontierEntry(program(entry.semantics()),
                                         logPrior=g.logLikelihood(request, program(entry.semantics())),
                                         logLikelihood=0.0)
                           for entry in lex._entries[key]]

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

  Returns:
    lexicon: Modified copy of the provided lexicon instance, with semantic
      entries modified to use new inventions.
    affected_entries: A dict mapping from invention names to the `Token`
      instances affected by (modified according to) the corresponding
      invention.
  """
  """
  WARNING!!!
  The code below assumes that compression does not reorder the FrontierEntry's of a Frontier.
  This might not be accurate.
  Code may break if there are multiple entries per frontier
  """

  lex = old_lex.clone()
  affected = defaultdict(list)

  for frontier in frontiers:
    word = frontier.task.name
    lex._entries[word] = []
    for frontier_entry, lex_entry in zip(frontier.entries, old_lex._entries[word]):
      raw_program_str = frontier_entry.program.show(False)

      involved_inventions = []

      # NB: it is important that `invented_name_dict` be ordered by
      # decreasing depth of the invented expressions.
      #
      # This way we ensure we attempt replacing with the largest
      # inventions first. Otherwise we might first replace a
      # sub-invention's expression and thereby kill our chance of later
      # replacing the containing invention's expression.
      for name in invented_name_dict:
        old_program_str = raw_program_str
        raw_program_str = raw_program_str.replace(invented_name_dict[name], name)

        if old_program_str != raw_program_str:
          # This lexical entry was affected by this invention.
          involved_inventions.append(name)

      semantics, _ = read_ec_sexpr(raw_program_str)
      #print("semantics", semantics)
      token = Token(word, lex_entry.categ(), semantics, lex_entry.weight())

      for inv_name in involved_inventions:
        affected[inv_name].append(token)

      lex._entries[word].append(token)

  return lex, dict(affected)


class Compressor(object):
  """
  Bridge between lexicon learning and EC compression.

  Provides an endpoint for searching for new inventions, and for carrying
  EC state between searches.
  """

  def __init__(self, ontology, **EC_kwargs):
    self.ontology = ontology
    # EC `Grammar` instance.
    self.grammar = ontology_to_grammar_initial(ontology)

    self.EC_kwargs = EC_kwargs

    self.invented_name_dict = None

  def make_inventions(self, lexicon):
    """
    Run compression on the given grammar/lexicon and attempt to create new inventions.
    Inventions will be added directly to this instance's `ontology`.

    Args:
      lexicon: `Lexicon` instance

    Returns:
      lexicon: new `Lexicon`, a modified copy of the original
      affected_entries: A dict mapping from invention names to the `Token`
        instances affected by (i.e. modified according to) the corresponding
        invention.
    """

    # TODO(max) document
    frontiers = extract_frontiers_from_lexicon(lexicon, self.grammar,
                                               invented_name_dict=self.invented_name_dict)

    # Induce new inventions using the present grammar and frontiers.
    self.grammar, new_frontiers = induceGrammar(self.grammar, frontiers, **self.EC_kwargs)

    # Convert result back to an ontology, switching to a naming scheme that
    # plays nice with our client's setup.
    self.ontology, self.invented_name_dict = grammar_to_ontology(self.grammar, self.ontology)

    lexicon, affected_entries = frontiers_to_lexicon(new_frontiers, lexicon, self.invented_name_dict)

    return lexicon, affected_entries

