from nose.tools import *

from nltk.sem.logic import Expression

from clevros.logic import *


def test_as_ec_sexpr():
  expr = Expression.fromstring(r"\x y z.foo(bar(x,y),baz(y,z),blah)")
  eq_(as_ec_sexpr(expr), "(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")


def test_read_ec_sexpr():
  expr = read_ec_sexpr("(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")
  eq_(expr, Expression.fromstring(r"\a b c.foo(bar(a,b),baz(b,c),blah)"))
