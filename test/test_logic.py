from nose.tools import *

from nltk.sem.logic import Expression, Variable, \
    FunctionVariableExpression, AndExpression, NegatedExpression

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


def _make_simple_mock_ontology():
  types = TypeSystem(["boolean", "obj"])
  functions = [
      types.new_function("and_", ("boolean", "boolean", "boolean"), lambda x, y: x and y),
      types.new_function("foo", ("obj", "boolean"), lambda x: True),
      types.new_function("bar", ("obj", "boolean"), lambda x: True),

      types.new_function("invented_1", (("obj", "boolean"), "obj", "boolean"), lambda f, x: x is not None and f(x)),

      types.new_function("threeplace", ("obj", "obj", "boolean", "boolean"), lambda x, y, o: True),
  ]
  constants = [types.new_constant("baz", "boolean")]

  ontology = Ontology(types, functions, constants, variable_weight=0.1)
  return ontology


def test_extract_lambda():
  """
  `extract_lambda` should support all possible orderings of the variables it
  encounters.
  """
  expr = Expression.fromstring(r"foo(\a.a,\a.a)")
  extracted = extract_lambda(expr)
  eq_(len(extracted), 2)


def test_iter_expressions():
  ontology = _make_simple_mock_ontology()
  from pprint import pprint

  cases = [
    (4, "Reuse of bound variable", (r"\z1.and_(foo(z1),bar(z1))",)),
    (3, "Support passing functions as arguments to higher-order functions",
     (r"\z1.invented_1(foo,z1)",)),
    (3, "Consider both argument orders",
     (r"\z1 z2.and_(z1,z2)", r"\z2 z1.and_(z1,z2)")),
    (3, "Consider both argument orders for three-place function",
     (r"\z1 z2.threeplace(z1,z2,baz)", r"\z2 z1.threeplace(z1,z2,baz)")),
  ]

  def do_case(max_depth, msg, assert_in):
    expressions = set(ontology.iter_expressions(max_depth=max_depth))
    expression_strs = list(map(str, expressions))

    present = [expr in expression_strs for expr in assert_in]
    pprint(list(zip(assert_in, present)))
    ok_(all(present), msg)

  for max_depth, msg, assert_in in cases:
    yield do_case, max_depth, msg, assert_in


def test_as_ec_sexpr():
  ont = _make_mock_ontology()
  expr = Expression.fromstring(r"\x y z.foo(bar(x,y),baz(y,z),blah)")
  eq_(ont.as_ec_sexpr(expr), "(lambda (lambda (lambda (foo (bar $2 $1) (baz $1 $0) blah))))")


def test_as_ec_sexpr_function():
  ont = _make_mock_ontology()
  expr = FunctionVariableExpression(Variable("and_", ont.types["boolean", "boolean", "boolean"]))
  eq_(ont.as_ec_sexpr(expr), "(lambda (lambda (and_ $1 $0)))")


def test_as_ec_sexpr_event():
  types = TypeSystem(["obj"])
  functions = [
    types.new_function("e", ("v",), lambda: ()),
    types.new_function("result", ("v", "obj"), lambda e: e),
  ]
  constants = []

  ontology = Ontology(types, functions, constants)

  cases = [
    (r"result(e)", "(result e)"),
    (r"\x.foo(x,e)", "(lambda (foo $0 e))"),
    (r"\x.foo(e,x)", "(lambda (foo e $0))"),
    (r"\x.foo(x,e,x)", "(lambda (foo $0 e $0))"),
    (r"\a.constraint(ltzero(cmp_pos(ax_z,pos,e,a)))", "(lambda (constraint (ltzero (cmp_pos ax_z pos e $0))))"),
  ]

  def do_case(expr, expected):
    expr = Expression.fromstring(expr)
    eq_(ontology.as_ec_sexpr(expr), expected)

  for expr, expected in cases:
    yield do_case, expr, expected


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
