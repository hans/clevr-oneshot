from nose.tools import *

from nltk.sem.logic import Expression

from clevros.logic import Ontology, TypeSystem, Function
from clevros.model import *


def test_model_constants():
  """
  Test evaluating with constant values.
  """
  types = TypeSystem(["num"])

  functions = [
    types.new_function("add", ("num", "num", "num"), lambda a, b: str(int(a) + int(b)))
  ]
  constants = [types.new_constant("1", "num"), types.new_constant("2", "num")]

  ontology = Ontology(types, functions, constants)
  model = Model(scene={"objects": []}, ontology=ontology)

  cases = [
    ("Test basic constant evaluation", r"1", "1"),
    ("Test constants as arguments to functions", r"add(1,1)", "2"),
  ]

  def test(msg, expr, expected):
    print("ret", model.evaluate(Expression.fromstring(expr)))
    eq_(model.evaluate(Expression.fromstring(expr)), expected, msg=msg)

  for msg, expr, expected in cases:
    yield test, msg, expr, expected


def test_model_induced_functions():
  """
  Test evaluating a model with an ontology which has induced functions.
  """

  fake_scene = {
    "objects": ["foo", "bar"],
  }

  types = TypeSystem(["a"])
  functions = [
      types.new_function("test", ("a", "a"), lambda x: True),
      types.new_function("test2", ("a", "a"), Expression.fromstring(r"\x.test(test(x))")),
  ]
  ontology = Ontology(types, functions, [])

  model = Model(scene=fake_scene, ontology=ontology)

  cases = [
    ("Test basic call of an abstract function", r"\a.test2(a)", {"foo": True, "bar": True}),
    ("Test embedded call of an abstract function", r"\a.test(test2(a))", {"foo": True, "bar": True}),
  ]

  def test(msg, expr, expected):
    eq_(model.evaluate(Expression.fromstring(expr)), expected, msg=msg)

  for msg, expr, expected in cases:
    yield test, msg, expr, expected
