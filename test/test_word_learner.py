
import copy

from nose.tools import *

from clevros.compression import Compressor
from clevros.lexicon import Lexicon
from clevros.logic import Ontology, TypeSystem
from clevros.word_learner import *


EC_kwargs = {
  "topK": 1,
  "pseudoCounts": 1.0,
  "a": 1,
  "aic": -1.0,
  "structurePenalty": 0.001,
  "backend": "pypy",
  "CPUs": 1,
}


def test_cyclic_derived_categories():
  types = TypeSystem(["obj", "boolean"])
  functions = [
    types.new_function("book", ("obj", "boolean"), lambda x: True),
    types.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
    types.new_function("unique", ("obj", ("obj", "boolean"), "obj"), lambda _, f: "foo"),
    types.new_function("id", ("obj", "obj"), lambda x: x),
  ]
  constants = [types.new_constant("obj1", "obj")]

  ontology = Ontology(types, functions, constants)


  # lexicon1: sexpr for N is more heavily nested than sexpr for S
  lexicon1 = Lexicon.fromstring(r"""
  :- S, N

  the => S/N {\x.unique(obj1,x)}
  that => S/N {\x.unique(obj1,x)}

  book => N {\x.and_(book(x),book(x))}
  books => N {\x.and_(book(x),book(x))}
  """, copy.deepcopy(ontology), include_semantics=True)

  # lexicon2: sexpr for S is more heavily nested than sexpr for N.
  # This should not affect how categories are propagated (even though it will
  # affect the ordering).
  lexicon2 = Lexicon.fromstring(r"""
  :- S, N

  the => S/N {\x.unique(id(id((id(id(obj1))))),x)}
  that => S/N {\x.unique(id(id((id(id(obj1))))),x)}

  book => N {\x.and_(book(x),book(x))}
  books => N {\x.and_(book(x),book(x))}
  """, copy.deepcopy(ontology), include_semantics=True)


  def do_case(lexicon, i):
    learner = WordLearner(lexicon, Compressor(lexicon.ontology, **EC_kwargs))
    learner.compress_lexicon()

    derived_S = \
        next(iter(learner.lexicon._derived_categories_by_base[learner.lexicon.parse_category("S")]))
    derived_N = \
        next(iter(learner.lexicon._derived_categories_by_base[learner.lexicon.parse_category("N")]))

    eq_(set(str(entry.categ()) for entry in learner.lexicon._entries["the"]),
        {"(%s/N)" % derived_S, "(S/%s)" % derived_N,
         "(%s/%s)" % (derived_S, derived_N), "(S/N)"})


  for i, lexicon in enumerate([lexicon1, lexicon2]):
    yield do_case, lexicon, str(i + 1)
