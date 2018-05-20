from nose.tools import *

from clevros.combinator import *
from clevros.lexicon import Lexicon

from nltk.ccg.api import FunctionalCategory, PrimitiveCategory
from nltk.sem.logic import Expression


def _make_dummy_lexicon():
  return Lexicon.fromstring(r"""
      :- A, B, C, D
      """)

def test_positional_forward_raise():
  lex = _make_dummy_lexicon()
  cases = [
    (0, "A/B", "C", "((A/(B/C))/C)"),
    (0, "A/B/C", "D", "(((A/B)/(C/D))/D)"),
  ]

  def do_case(index, left, right, expected):
    pfr = PositionalForwardRaiseCombinator(index)
    left = lex.parse_category(left)
    right = lex.parse_category(right)

    ok_(pfr.can_combine(left, right))
    eq_(str(next(iter(pfr.combine(left, right)))), expected)

  for index, left, right, expected in cases:
    yield do_case, index, left, right, expected


def test_positional_forward_raise_semantics():
  lex = _make_dummy_lexicon()

  cases = [
    (0, "A/B", r"\b.foo(bar,b,baz)", "C", "((A/(B/C))/C)", r"\z1 F1.foo(bar,F1(z1),baz)"),
  ]

  def do_case(index, left, semantics, right, expected_synt, expected_sem):
    pfr = PositionalForwardRaiseCombinator(index)
    left = lex.parse_category(left)
    right = lex.parse_category(right)

    ok_(pfr.can_combine(left, right))
    eq_(str(next(iter(pfr.combine(left, right)))), expected_synt)

    semantics = Expression.fromstring(semantics)
    expected_sem = str(Expression.fromstring(expected_sem).simplify())
    eq_(str(pfr.update_semantics(semantics)), expected_sem)

  for index, left, semantics, right, expected_synt, expected_sem in cases:
    yield do_case, index, left, semantics, right, expected_synt, expected_sem


def test_category_search_replace():
  lex = _make_dummy_lexicon()

  cases = [
      ("A/B", "B", "C", ["A/C"]),
      ("A/B", "A", "C", ["C/B"]),
      ("A/A", "A", "B", ["A/B", "B/A", "B/B"]),

      ("A/(B/C)", "B/C", "D", ["A/D"]),
      ("(A/B)/C", "B/C", "D", []),
  ]

  def do_case(expr, search, replace, expected):
    eq_(category_search_replace(lex.parse_category(expr), lex.parse_category(search),
                                lex.parse_category(replace)),
        set(lex.parse_category(expected_i) for expected_i in expected))

  for expr, search, replace, expected in cases:
    yield do_case, expr, search, replace, expected


def test_type_raised_category_search_replace():
  lex = _make_dummy_lexicon()

  cases = [
      # All basic cases should still pass.
      ("A/B", "B", "C", ["A/C"]),
      ("A/B", "A", "C", ["C/B"]),
      ("A/A", "A", "B", ["A/B", "B/A", "B/B"]),

      ("A/(B/C)", "B/C", "D", ["A/D", "((A/(D/C))/C)"]), # new PFR result on this one
      ("(A/B)/C", "B/C", "D", []),
  ]

  def do_case(expr, search, replace, expected):
    eq_(type_raised_category_search_replace(lex.parse_category(expr), lex.parse_category(search),
                                            lex.parse_category(replace)),
        set(lex.parse_category(expected_i) for expected_i in expected))

  for expr, search, replace, expected in cases:
    yield do_case, expr, search, replace, expected
