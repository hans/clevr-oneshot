from nose.tools import *

from clevros.combinator import *
from clevros.lexicon import Lexicon

from nltk.ccg.api import FunctionalCategory, PrimitiveCategory
from nltk.sem.logic import Expression


def test_positional_forward_raise():
  lex = Lexicon.fromstring(r"""
      :- A, B, C, D
      """)

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
  lex = Lexicon.fromstring(r"""
      :- A, B, C
      """, include_semantics=True)

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
