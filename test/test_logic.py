from nose.tools import *

from nltk.sem.logic import Expression

from clevros.logic import *


def _make_mock_ontology():
  def fn_unique(xs):
    true_xs = [x for x, matches in xs.items() if matches]
    assert len(true_xs) == 1
    return true_xs[0]

  types = set(["obj", "num", "ax", "boolean", ("obj", "boolean")])

  functions = [
    Function("cmp_pos", ("ax", "obj", "obj", "num"),
            lambda ax, a, b: a["3d_coords"][ax()] - b["3d_coords"][ax()]),
    Function("ltzero", ("num", "boolean"), lambda x: x < 0),

    Function("ax_x", ("ax",), lambda: 0),
    Function("ax_y", ("ax",), lambda: 1),
    Function("ax_z", ("ax",), lambda: 2),

    Function("unique", (("obj", "boolean"), "obj"), fn_unique),

    Function("cube", ("obj", "boolean"), lambda x: x["shape"] == "cube"),
    Function("sphere", ("obj", "boolean"), lambda x: x["shape"] == "sphere"),
  ]

  ontology = Ontology(types, functions, variable_weight=0.1)

  return ontology


def test_as_ec_sexpr():
  expr = Expression.fromstring(r"\x y z.foo(bar(x,y),baz(y,z),blah)")
  eq_(as_ec_sexpr(expr), "(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")


def test_read_ec_sexpr():
  expr = read_ec_sexpr("(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")
  eq_(expr, Expression.fromstring(r"\a b c.foo(bar(a,b),baz(b,c),blah)"))


def test_valid_lambda_expr():
  """
  Regression test: valid_lambda_expr was rejecting this good sub-expression at c720b4
  """
  ontology = _make_mock_ontology()
  eq_(ontology._valid_lambda_expr(Expression.fromstring(r"\b.ltzero(cmp_pos(ax_x,a,b))"), bound_vars=()), False)
  eq_(ontology._valid_lambda_expr(Expression.fromstring(r"\b.ltzero(cmp_pos(ax_x,a,b))"), bound_vars=(('a', None),)), True)
