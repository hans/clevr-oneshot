from nose.tools import *

from nltk.sem.logic import Expression, Variable

from clevros.logic import *


def _make_mock_ontology():
  def fn_unique(xs):
    true_xs = [x for x, matches in xs.items() if matches]
    assert len(true_xs) == 1
    return true_xs[0]

  types = TypeSystem(["obj", "num", "ax", "boolean"])

  functions = [
    types.new_function("cmp_pos", ("ax", "obj", "obj", "num"),
                       lambda ax, a, b: a["3d_coords"][ax()] - b["3d_coords"][ax()]),
    types.new_function("ltzero", ("num", "boolean"), lambda x: x < 0),

    types.new_function("ax_x", ("ax",), lambda: 0),
    types.new_function("ax_y", ("ax",), lambda: 1),
    types.new_function("ax_z", ("ax",), lambda: 2),

    types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),

    types.new_function("cube", ("obj", "boolean"), lambda x: x["shape"] == "cube"),
    types.new_function("sphere", ("obj", "boolean"), lambda x: x["shape"] == "sphere"),
  ]

  ontology = Ontology(types, functions, variable_weight=0.1)

  return ontology


def test_as_ec_sexpr():
  expr = Expression.fromstring(r"\x y z.foo(bar(x,y),baz(y,z),blah)")
  eq_(as_ec_sexpr(expr), "(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")


def test_read_ec_sexpr():
  expr, bound_vars = read_ec_sexpr("(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")
  eq_(expr, Expression.fromstring(r"\a b c.foo(bar(c,b),baz(b,a),blah)"))


def test_read_ec_sexpr_de_bruijn():
  """
  properly handle de Bruijn indexing in EC lambda expressions.
  """
  expr, bound_vars = read_ec_sexpr("(lambda ((lambda ($0 (lambda $0))) (lambda ($1 $0))))")
  print(expr)
  eq_(expr, Expression.fromstring(r"\A.((\B.B(\C.C))(\C.A(C)))"))


def test_read_ec_sexpr_nested():
  """
  read_ec_sexpr should support reading in applications where the function
  itself is an expression (i.e. there is some not-yet-reduced beta reduction
  candidate).
  """
  expr, bound_vars = read_ec_sexpr("(lambda ((lambda (foo $0)) $0))")
  eq_(expr, Expression.fromstring(r"\a.((\b.foo(b))(a))"))


def test_read_ec_sexpr_higher_order_param():
  expr, bound_vars = read_ec_sexpr("(lambda (lambda ($0 $1)))")
  eq_(expr, Expression.fromstring(r"\a P.P(a)"))


def test_valid_lambda_expr():
  """
  Regression test: valid_lambda_expr was rejecting this good sub-expression at c720b4
  """
  ontology = _make_mock_ontology()
  eq_(ontology._valid_lambda_expr(Expression.fromstring(r"\b.ltzero(cmp_pos(ax_x,a,b))"), ctx_bound_vars=()), False)
  eq_(ontology._valid_lambda_expr(Expression.fromstring(r"\b.ltzero(cmp_pos(ax_x,a,b))"), ctx_bound_vars=(Variable('a'),)), True)


def test_typecheck():
  ontology = _make_mock_ontology()

  def do_test(expr, extra_signature):
    expr = Expression.fromstring(expr)
    ontology.typecheck(expr, extra_signature)
    print(expr.type)

  exprs = [
      (r"ltzero(cmp_pos(ax_x,unique(\x.sphere(x)),unique(\y.cube(y))))",
       {"x": ontology.types["obj"], "y": ontology.types["obj"]}),
      (r"\a b.ltzero(cmp_pos(ax_x,a,b))",
       {"a": ontology.types["obj"], "b": ontology.types["obj"]}),
  ]

  for expr, extra_signature in exprs:
    yield do_test, expr, extra_signature


def test_infer_type():
  ontology = _make_mock_ontology()

  def do_test(expr, query_variable, expected_type):
    eq_(ontology.infer_type(Expression.fromstring(expr), query_variable), expected_type)

  cases = [
    (r"\a.sphere(a)", "a", ontology.types["obj"]),
    (r"\a.ltzero(cmp_pos(ax_x,a,a))", "a", ontology.types["obj"]),
    (r"\a b.ltzero(cmp_pos(ax_x,a,b))", "a", ontology.types["obj"]),
    (r"\a b.ltzero(cmp_pos(ax_x,a,b))", "b", ontology.types["obj"]),
  ]

  for expr, query_variable, expected_type in cases:
    yield do_test, expr, query_variable, expected_type
