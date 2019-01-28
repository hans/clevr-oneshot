from nose.tools import *

from nltk.sem.logic import Expression

from clevros.lexicon import Lexicon
from clevros.util import UniquePriorityQueue, ConditionalDistribution, Distribution


def test_unique_priority_queue():
  q = UniquePriorityQueue(maxsize=10)

  lex = Lexicon.fromstring(r"""
      :- S, N

      the => S/N
      """)

  item = (22/7, (lex.parse_category("S/N/N"), Expression.fromstring(r"\z1 z2.foo(z1,z2)")))
  q.put_nowait(item)
  q.put_nowait(item)

  eq_(q.qsize(), 1)


def test_regression_normalize_all():
  cond_dist = ConditionalDistribution()
  cond_dist.dists.update({
    r"(N/N)": Distribution({'unique': 0.002, 'become': 0.001, 'move': 0.001, 'toy': 0.001, 'cause': 0.001, 'not': 0.001, 'contact': 0.001, 'nott': 0.001, 'shade': 0.001, 'female': 0.001, None: 0.001}),
    r"N": Distribution({'female': 5.0686229734345556, 'toy': 0.0033367507102867898, 'shade': 0.002, 'become': 0.001, 'move': 0.001, 'cause': 0.001, 'contact': 0.001, 'nott': 0.001, None: 0.001, 'not': -0.99441764432088586, 'unique': -3.0689597241448427}),
    r"((S\N)/(S\N))": Distribution({'become': 0.001, 'move': 0.001, 'unique': 0.001, 'toy': 0.001, 'cause': 0.001, 'not': 0.001, 'contact': 0.001, 'nott': 0.001, 'shade': 0.001, 'female': 0.001, None: 0.001}),
    r"((S\N)/N)": Distribution({'contact': 0.32708695652173914, 'become': 0.022739130434782619, 'cause': 0.022739130434782619, 'move': 0.001, 'unique': 0.001, 'toy': 0.001, 'not': 0.001, 'nott': 0.001, 'shade': 0.001, 'female': 0.001, None: 0.001}),
    r"(S\N)": Distribution({'nott': 0.010601691596803159, 'contact': 0.010601691596803159, 'move': 0.0081040406235274448, 'become': 0.001, 'unique': 0.001, 'cause': 0.001, 'not': 0.001, 'shade': 0.001, 'female': 0.001, None: 0.001, 'toy': -0.98479191875294514})})

  cond_dist.normalize_all()
  for key in cond_dist.dists:
    ok_(cond_dist.dists[key] is not None, key)
