from clevros.lexicon import Lexicon
from clevros.logic import TypeSystem, Ontology
from clevros.primitives import *


types = TypeSystem(["obj", "boolean", "manner", "action", "set", "str"])

functions = [
  # Logical operation
  types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),
  types.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
  types.new_function("not_", ("boolean", "boolean"), lambda a: not a),
  types.new_function("eq", ("?", "?", "boolean"), lambda a, b: a == b),

  # Ops on objects
  types.new_function("book", ("obj", "boolean"), lambda x: x["type"] == "book"),
  # types.new_function("apple", ("obj", "boolean"), lambda x: x["type"] == "apple"),
  # types.new_function("cookie", ("obj", "boolean"), lambda x: x["type"] == "cookie"),
  types.new_function("water", ("obj", "boolean"), lambda x: x["substance"] == "water"),
  types.new_function("paint", ("obj", "boolean"), lambda x: x["substance"] == "paint"),
  types.new_function("wall", ("obj", "boolean"), lambda x: x["type"] == "wall"),
  types.new_function("table", ("obj", "boolean"), lambda x: x["type"] == "table"),
  types.new_function("jar", ("obj", "boolean"), lambda x: x["type"] == "jar"),
  types.new_function("orientation", ("obj", "manner"), lambda x: x["orientation"]),
  types.new_function("liquid", ("obj", "boolean"), lambda x: x["state"] == "liquid"),
  types.new_function("full", ("obj", "boolean"), lambda x: x["full"]),
  # Two-place ops on objects
  types.new_function("contact", ("obj", "obj", "boolean"), fn_contact),
  types.new_function("contain", ("obj", "obj", "boolean"), fn_contain),

  # Ops on sets
  types.new_function("characteristic", ("set", "str", "boolean"),
                           lambda s, fn: s.characteristic == fn),

  # Ops on events
  types.new_function("e", ("v",), Event()),
  types.new_function("direction", ("v", "manner"), lambda e: e.direction),
  types.new_function("result", ("v", "obj"), lambda e: e.result),
  types.new_function("patient", ("v", "obj"), lambda e: e.patient),

  # Actions
  types.new_function("put", ("v", "obj", "manner", "action"), Put),
  types.new_function("constraint", ("boolean", "manner"), Constraint), # TODO funky type
  types.new_function("addc", ("manner", "manner", "manner"), lambda c1, c2: Constraint(c1, c2)),
  types.new_function("join", ("action", "boolean", "action"), ActAndEntail),
]


constants = [
  types.new_constant("vertical", "manner"),
  types.new_constant("horizontal", "manner"),
  types.new_constant("up", "manner"),
  types.new_constant("down", "manner"),

  types.new_constant("apple", "str"),
  types.new_constant("cookie", "str"),
]

ontology = Ontology(types, functions, constants)


lexicon = Lexicon.fromstring(r"""
  :- S, N, PP

  the => N/N {\x.unique(x)}

  book => N {\x.book(x)}
  water => N {\x.water(x)}
  paint => N {\x.paint(x)}
  wall => N {\x.wall(x)}
  jar => N {\x.jar(x)}
  apples => N {\x.characteristic(x,apple)}
  cookies => N {\x.characteristic(x,cookie)}

  on => PP/N {\a.constraint(contact(a,result(e)))}
  onto => PP/N {\a.constraint(contact(a,result(e)))}

  put => S/N/PP {\d o.put(e,o,d)}
  set => S/N/PP {\d o.put(e,o,d)}

  # "hang the picture on the wall"
  hang => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(orientation(result(e)),vertical))))}
  # TODO: need to allow that a single wordform have multiple possible syntactic arities..
  hangs => S/N {\o.put(e,o,constraint(eq(orientation(result(e)),vertical)))}
  lay => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(orientation(result(e)),horizontal))))}

  drop => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(direction(e),down))))}
  hoist => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(direction(e),up))))}

  pour => S/N/PP {\d o.put(e,o,addc(d,constraint(liquid(result(e)))))}
  spill => S/N/PP {\d o.put(e,o,addc(d,constraint(liquid(result(e)))))}

  # "spray the wall with paint"
  spray => S/N/PP {\d o.put(e,o,addc(d,constraint(not_(full(result(e))))))}
  load => S/N/PP {\d o.put(e,o,addc(d,constraint(not_(full(result(e))))))}

  # "fill the jar with cookies"
  fill => S/N/PP {\o d.put(e,o,addc(constraint(contain(d,result(e))),constraint(full(result(e)))))}
  stuff => S/N/PP {\o d.put(e,o,addc(constraint(contain(d,result(e))),constraint(full(result(e)))))}
  """, ontology, include_semantics=True)


scene = {
  "objects": [
    frozendict({"type": "jar"}),
    frozendict({"type": "book"}),
    frozendict({"type": "table"}),
  ]
}

event = Event()

examples = [
  # # target:
  # # \d.put(e,result(e),addc(constraint(contain(d,result(e))),constraint(full(result(e)))))
  # # ==> \d.invented_0(addc(...),d,result(e))
  # # where invented_0 = \a b c.put(e,c,addc(b,constraint(a)))
  # ("fill the jar", scene, Put(event, event.result, Constraint(event.patient.contains(event.result),event.patient.full))),

  ("put the book on the table", scene, True),
]

