from nose.tools import *

from clevros.lexicon import *

from nltk.ccg.lexicon import FunctionalCategory, PrimitiveCategory, Direction


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


def test_get_semantic_arity():
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
    eq_(get_semantic_arity(augParseCategory(cat, lex._primitives, lex._families)[0]),
        expected, msg=str(cat))

  for cat, expected in cases:
    yield test_case, cat, expected


def _make_lexicon_with_derived_category():
  lex = Lexicon.fromstring(r"""
  :- S, NP

  the => S/NP {\x.unique(x)}

  # Have an entry taking the same argument type twice.
  derp => S/NP/NP {\a b.derp(a,b)}

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

  eq_(len(lex._entries["the"]), 2,
      msg="Derived category propagation should have created a new functional "
          "category entry for the higher-order `the`. Only %i entries." % len(lex._entries["the"]))

  # Should try one of each possible replacement with a derived category,
  # yielding 3 entries for derp
  eq_(set(str(entry.categ()) for entry in lex._entries["derp"]),
      set(["((S/NP)/NP)", "((S/NP)/D0{NP})", "((S/D0{NP})/NP)"]))


def test_get_candidate_derived():
  """
  `get_candidate_categories` should provide special treatment to derived
  categories.
  """

  lex, involved_tokens, cat_name = _make_lexicon_with_derived_category()
  lex.propagate_derived_category(cat_name)

  test_sentence = "a foo".split()
  tokens = ["a"]

  expected = {lex._entries["the"][0].categ(),
              FunctionalCategory(PrimitiveCategory("S"), lex._derived_categories[cat_name][0], Direction('/', ('', '')))}
  eq_(get_candidate_categories(lex, tokens, test_sentence)["a"], expected)


def test_get_lf_unigrams():
  lex = Lexicon.fromstring(r"""
    :- NN

    the => NN/NN {\x.unique(x)}
    sphere => NN {\x.and_(object(x),sphere(x))}
    cube => NN {\x.and_(object(x),cube(x))}
    """, include_semantics=True)

  expected = {
    "NN": Counter({"and_": 2 / 6, "object": 2 / 6, "sphere": 1 / 6, "cube": 1 / 6}),
    "(NN/NN)": Counter({"unique": 1})
  }

  ngrams = lex.lf_ngrams(order=1, condition_on_syntax=True, smooth=False)
  for categ, counter in ngrams.items():
    eq_(counter, expected[str(categ)])


def test_get_yield():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- S, NN, PP

    on => PP/NN
    the => S/NN
    the => NN/NN
    sphere => NN
    sphere => NN
    """)

  cases = [
      ("S", "S"),
      ("S/NN", "S"),
      (r"NN\PP/NN", "PP"),
      (r"NN/(PP/NN)", "NN"),
  ]

  def test_case(cat, cat_yield):
    eq_(get_yield(augParseCategory(cat, lex._primitives, lex._families)[0]), cat_yield)

  for cat, cat_yield in cases:
    yield test_case, cat, cat_yield


def test_propagate_functional_category():
  """
  Validate that functional categories are correctly propagated.
  """

  # This is very tricky! Suppose have a derived functional category `X/Y` and
  # there are other entries `S/X`. After propagation, we want there to be some
  # explicit type lifted form `S/(D0/Y)` where `D0 = (X/Y)`.
  lex = Lexicon.fromstring(r"""
  :- S, NN, PP

  put => S/NN/PP
  it => NN
  on => PP/NN
  the_table => NN
  """)

  involved_tokens = [lex._entries["on"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)
  lex.propagate_derived_category(derived_categ)

  eq_(set(str(entry.categ()) for entry in lex._entries["put"]),
      set(["((S/NN)/PP)", "(((S/NN)/%s)/NN)" % lex._derived_categories[derived_categ][0]]))


