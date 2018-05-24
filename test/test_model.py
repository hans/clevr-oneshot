from nose.tools import *

from frozendict import frozendict
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

  ontology = Ontology(types, functions, constants, add_default_functions=False)
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
  ontology = Ontology(types, functions, [], add_default_functions=False)

  model = Model(scene=fake_scene, ontology=ontology)

  cases = [
    ("Test basic call of an abstract function", r"\a.test2(a)", {"foo": True, "bar": True}),
    ("Test embedded call of an abstract function", r"\a.test(test2(a))", {"foo": True, "bar": True}),
  ]

  def test(msg, expr, expected):
    eq_(model.evaluate(Expression.fromstring(expr)), expected, msg=msg)

  for msg, expr, expected in cases:
    yield test, msg, expr, expected


def test_model_complex():
  """
  Test evaluating some complex expressions (which yield Python objects as results).
  """

  from clevros import primitives as p

  scene = {
    "objects": [
      frozendict({"female": True, "agent": True, "shape": "person"}),
      frozendict({"shape": "donut"}),
      frozendict({"shape": "cube"}),
    ]
  }
  examples = [
    ("gorp the female the cube", scene,
     p.ComposedAction(p.CausePossession(scene["objects"][0], {"shape": "cube"}), p.Transfer({"shape": "cube"}, scene["objects"][0], "far"))),
  ]

  types = TypeSystem(["obj", "num", "ax", "dist", "boolean", "action"])

  functions = [
    types.new_function("and_", ("boolean", "boolean", "boolean"), p.fn_and),

    types.new_function("unique", (("obj", "boolean"), "obj"), p.fn_unique),

    types.new_function("cube", ("obj", "boolean"), p.fn_cube),

    types.new_function("male", ("obj", "boolean"), lambda x: x.get("male", False)),
    types.new_function("female", ("obj", "boolean"), lambda x: x.get("female", False)),

    types.new_function("object", (types.ANY_TYPE, "boolean"), p.fn_object),
    types.new_function("agent", (types.ANY_TYPE, "boolean"), lambda x: x.get("agent", False)),

    types.new_function("cause_possession", ("obj", "obj", "action"), lambda agent, obj: p.CausePossession(agent, obj)),
    types.new_function("transfer", ("obj", "obj", "dist", "action"), lambda obj, agent, dist: p.Transfer(obj, agent, dist)),

    types.new_function("do_", ("action", "action", "action"), lambda a1, a2: a1 + a2),
  ]

  constants = [types.new_constant("any", "dist"), types.new_constant("far", "dist"), types.new_constant("near", "dist")]

  ontology = Ontology(types, functions, constants, variable_weight=0.1)
  model = Model(scene, ontology)

  eq_(model.evaluate(Expression.fromstring(r"cause_possession(unique(\a.female(a)),unique(\b.and_(object(b),cube(b))))")),
      p.CausePossession(scene["objects"][0], scene["objects"][2]))

  eq_(model.evaluate(Expression.fromstring(r"do_(cause_possession(unique(\a.female(a)),unique(\b.and_(object(b),cube(b)))),transfer(unique(\c.and_(object(c),cube(c))),unique(\d.female(d)),any))")),
      p.ComposedAction(p.CausePossession(scene["objects"][0], scene["objects"][2]),
                       p.Transfer(scene["objects"][2], scene["objects"][0], "any")))
