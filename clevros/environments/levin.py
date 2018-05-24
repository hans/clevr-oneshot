from clevros.lexicon import Lexicon
from clevros.logic import TypeSystem, Ontology
from clevros.primitives import *


types = TypeSystem(["obj", "boolean", "manner", "action", "set", "str"])

functions = [
  # Logical operation
  types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),
  types.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
  types.new_function("eq", ("?", "?", "boolean"), lambda a, b: a == b),

  # Ops on objects
  types.new_function("book", ("obj", "boolean"), lambda x: x["type"] == "book"),
  # types.new_function("apple", ("obj", "boolean"), lambda x: x["type"] == "apple"),
  # types.new_function("cookie", ("obj", "boolean"), lambda x: x["type"] == "cookie"),
  types.new_function("water", ("obj", "boolean"), lambda x: x["substance"] == "water"),
  types.new_function("paint", ("obj", "boolean"), lambda x: x["substance"] == "paint"),
  types.new_function("orientation", ("obj", "manner"), lambda x: x["orientation"]),
  types.new_function("liquid", ("obj", "boolean"), lambda x: x["state"] == "liquid"),
  types.new_function("full", ("obj", "boolean"), lambda x: x["full"]),

  # Ops on sets
  types.new_function("characteristic", ("set", "str", "boolean"),
                           lambda s, fn: s.characteristic == fn),

  # Ops on events
  types.new_function("e", ("v",), Event()),
  types.new_function("direction", ("v", "manner"), lambda e: e.direction),
  types.new_function("result", ("v", "obj"), lambda e: e.result),

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

  apples => N {\x.characteristic(x,apple)}
  cookies => N {\x.characteristic(x,cookie)}

  put => S/N/PP {\d o.put(e,o,d)}
  set => S/N/PP {\d o.put(e,o,d)}

  # "hang the picture on the wall"
  hang => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(orientation(result(e)),vertical))))}
  lay => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(orientation(result(e)),horizontal))))}

  drop => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(direction(e),down))))}
  hoist => S/N/PP {\d o.put(e,o,addc(d,constraint(eq(direction(e),up))))}

  pour => S/N/PP {\d o.join(put(e,o,d),liquid(result(e)))}
  spill => S/N/PP {\d o.join(put(e,o,d),liquid(result(e)))}

  # "spray the wall with paint"
  spray => S/N/PP {\d o.put(e,o,addc(d,constraint(-full(result(e)))))}
  load => S/N/PP {\d o.put(e,o,addc(d,constraint(-full(result(e)))))}

  # "fill the jar with cookies"
  fill => S/N/PP {\o d.put(e,o,addc(d,constraint(full(result(e)))))}
  stuff => S/N/PP {\o d.put(e,o,addc(d,constraint(full(result(e)))))}
  """, ontology, include_semantics=True)


examples = [

]
