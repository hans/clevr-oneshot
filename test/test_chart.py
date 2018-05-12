from nose.tools import *

from clevros.chart import *
from clevros.lexicon import Lexicon


def _make_lexicon_with_derived_category():
  lex = Lexicon.fromstring(r"""
  :- S, NP

  the => S/NP {\x.unique(x)}

  foo => NP {\x.foo(x)}
  bar => NP {\x.bar(x)}
  baz => NP {\x.baz(x)}
  """, include_semantics=True)
  old_lex = lex.clone()

  # Induce a derived category involving `foo` and `bar`.
  involved_tokens = [lex._entries["foo"][0], lex._entries["bar"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)

  return old_lex, lex, involved_tokens, derived_categ


def test_parse_with_derived_category():
  """
  Ensure that we can still parse with derived categories.
  """

  old_lex, lex, involved_tokens, categ_name = _make_lexicon_with_derived_category()
  lex.propagate_derived_category(categ_name)

  old_results = WeightedCCGChartParser(old_lex).parse("the foo".split())
  results = WeightedCCGChartParser(lex).parse("the foo".split())

  eq_(len(results), len(results))
  eq_(results[0].label()[0].semantics(), old_results[0].label()[0].semantics())


def test_parse_oblique():
  """
  Test parsing a verb with an oblique PP -- this shouldn't require type raising?
  """

  lex = Lexicon.fromstring(r"""
  :- S, NP, PP

  place => S/NP/PP
  it => NP
  on => PP/NP
  the_table => NP
  """)

  parser = WeightedCCGChartParser(lex, ApplicationRuleSet)
  printCCGDerivation(parser.parse("place it on the_table".split())[0])
