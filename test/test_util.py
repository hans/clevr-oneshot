from nose.tools import *

from nltk.sem.logic import Expression

from clevros.lexicon import Lexicon
from clevros.util import UniquePriorityQueue


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
