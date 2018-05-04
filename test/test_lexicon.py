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


def _make_lexicon_with_derived_category():
  lex = Lexicon.fromstring(r"""
  :- S, NP

  the => S/NP {\x.unique(x)}

  foo => NP {\x.foo(x)}
  bar => NP {\x.bar(x)}
  baz => NP {\x.baz(x)}
  """, include_semantics=True)

  # Induce a derived category involving `foo` and `bar`.
  involved_tokens = [lex._entries["foo"][0], lex._entries["bar"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)

  return lex, involved_tokens, derived_categ


def test_propagate_derived_category():
  lex, involved_tokens, name = _make_lexicon_with_derived_category()
  assert name in lex._derived_categories

  old_baz_categ = lex._entries["baz"][0].categ()

  categ, _ = lex._derived_categories[name]

  lex.propagate_derived_category(name)

  eq_(lex._entries["foo"][0].categ(), categ)
  eq_(lex._entries["bar"][0].categ(), categ)
  eq_(lex._entries["baz"][0].categ(), old_baz_categ,
      msg="Propagation of derived category should not affect `baz`, which has a "
          "category which is the same as the base of the derived category")
