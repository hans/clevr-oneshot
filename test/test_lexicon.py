from nose.tools import *

from clevros.lexicon import *


def test_filter_lexicon_entry():
  lex = Lexicon.fromstring(r"""
    :- NN, DET, ADJ

    DET :: NN/NN
    ADJ :: NN/NN

    the => DET {\x.unique(x)}
    sphere => NN {filter_shape(scene,sphere)}
    sphere => NN {filter_shape(scene,cube)}
    """, include_semantics=True)

  lex_filtered = filter_lexicon_entry(lex, "sphere", "the sphere".split(), "unique(filter_shape(scene,sphere))")

  entries = lex_filtered.categories("sphere")
  assert len(entries) == 1

  eq_(str(entries[0].semantics()), "filter_shape(scene,sphere)")


def test_get_category_arity():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- NN, DET, ADJ

    DET :: NN/NN
    ADJ :: NN/NN

    the => DET {\x.unique(x)}
    sphere => NN {filter_shape(scene,sphere)}
    sphere => NN {filter_shape(scene,cube)}
    """, include_semantics=True)

  cases = [
      (r"NN", 0),
      (r"NN/NN", 1),
      (r"NN\NN", 1),
      (r"(NN\NN)/NN", 2),
  ]

  def test_case(cat, expected):
    eq_(get_category_arity(augParseCategory(cat, lex._primitives, lex._families)[0]),
        expected, msg=str(cat))

  for cat, expected in cases:
    yield test_case, cat, expected
