from nose.tools import *

from nltk.sem.logic import Expression, Variable, FunctionVariableExpression

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

    types.new_function("and_", ("boolean", "boolean", "boolean"), lambda x, y: x and y),
  ]

  constants = [types.new_constant("one", "num"), types.new_constant("two", "num")]

  ontology = Ontology(types, functions, constants, variable_weight=0.1)

  return ontology


def test_iter_expressions():
  """
  Functional expression iteration test involving higher-order functions.
  """

  types = TypeSystem(["boolean", "obj"])
  functions = [
      types.new_function("and_", ("boolean", "boolean", "boolean"), lambda x, y: x and y),
      types.new_function("foo", ("obj", "boolean"), lambda x: True),
      types.new_function("bar", ("obj", "boolean"), lambda x: True),

      types.new_function("invented_1", (("obj", "boolean"), "obj", "boolean"), lambda f, x: x is not None and f(x)),
  ]
  constants = []

  ontology = Ontology(types, functions, constants, variable_weight=0.1)

  expressions = list(ontology.iter_expressions(4))
  from pprint import pprint
  pprint(expressions)

  expression_strs = list(map(str, expressions))
  ok_(r"\z1.and_(foo(z1),bar(z1))" in expression_strs,
      "Reuse of bound variable")
  ok_(r"\z1.invented_1(foo,z1)" in expression_strs,
      "Support passing functions as arguments to higher-order functions")


def test_as_ec_sexpr():
  expr = Expression.fromstring(r"\x y z.foo(bar(x,y),baz(y,z),blah)")
  eq_(as_ec_sexpr(expr), "(lambda (lambda (lambda (foo (bar $2 $1) (baz $1 $0) blah))))")


def test_as_ec_sexpr_function():
  ont = _make_mock_ontology()
  expr = FunctionVariableExpression(Variable("F", ont.types["boolean", "boolean", "boolean"]))
  eq_(as_ec_sexpr(expr), "(lambda (lambda (F $1 $0)))")


def test_read_ec_sexpr():
  expr, bound_vars = read_ec_sexpr("(lambda (lambda (lambda (foo (bar $0 $1) (baz $1 $2) blah))))")
  eq_(expr, Expression.fromstring(r"\a b c.foo(bar(c,b),baz(b,a),blah)"))
  eq_(len(bound_vars), 3)


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

  def do_test(expr, extra_signature, expected):
    expr = Expression.fromstring(expr)
    ontology.typecheck(expr, extra_signature)
    eq_(expr.type, expected)

  exprs = [
      (r"ltzero(cmp_pos(ax_x,unique(\x.sphere(x)),unique(\y.cube(y))))",
       {"x": ontology.types["obj"], "y": ontology.types["obj"]},
       ontology.types["boolean"]),

      (r"\a b.ltzero(cmp_pos(ax_x,a,b))",
       {"a": ontology.types["obj"], "b": ontology.types["obj"]},
       ontology.types["obj", "obj", "boolean"]),

      (r"\A b.and_(ltzero(b),A(b))",
       {"A": ontology.types[ontology.types.ANY_TYPE, "boolean"], "b": ontology.types["num"]},
       ontology.types[(ontology.types.ANY_TYPE, "boolean"), "num", "boolean"]),
  ]

  for expr, extra_signature, expected in exprs:
    yield do_test, expr, extra_signature, expected


def test_infer_type():
  ontology = _make_mock_ontology()

  def do_test(expr, query_variable, expected_type):
    eq_(ontology.infer_type(Expression.fromstring(expr), query_variable), expected_type)

  cases = [
    (r"\a.sphere(a)", "a", ontology.types["obj"]),
    (r"\a.ltzero(cmp_pos(ax_x,a,a))", "a", ontology.types["obj"]),
    (r"\a b.ltzero(cmp_pos(ax_x,a,b))", "a", ontology.types["obj"]),
    (r"\a b.ltzero(cmp_pos(ax_x,a,b))", "b", ontology.types["obj"]),
    (r"\A b.and_(ltzero(b),A(b))", "A", ontology.types[ontology.types.ANY_TYPE, "boolean"]),
  ]

  for expr, query_variable, expected_type in cases:
    yield do_test, expr, query_variable, expected_type
