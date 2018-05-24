from clevros.logic import Ontology, TypeSystem
from clevros.primitives import *


types = TypeSystem(["obj", "num", "ax", "manner", "boolean", "action"])

functions = [
  types.new_function("cmp_pos", ("ax", "manner", "v", "obj", "num"), fn_cmp_pos),
  types.new_function("ltzero", ("num", "boolean"), fn_ltzero),
  types.new_function("and_", ("boolean", "boolean", "boolean"), fn_and),
  types.new_function("constraint", ("boolean", "manner"), lambda fn: Constraint(fn)),
  types.new_function("e", ("v",), Event()),

  types.new_function("ax_x", ("ax",), fn_ax_x),
  types.new_function("ax_y", ("ax",), fn_ax_y),
  types.new_function("ax_z", ("ax",), fn_ax_z),

  types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),

  types.new_function("cube", ("obj", "boolean"), fn_cube),
  types.new_function("sphere", ("obj", "boolean"), fn_sphere),
  types.new_function("donut", ("obj", "boolean"), fn_donut),
  types.new_function("pyramid", ("obj", "boolean"), fn_pyramid),
  types.new_function("hose", ("obj", "boolean"), fn_hose),
  types.new_function("cylinder", ("obj", "boolean"), fn_cylinder),

  types.new_function("letter", ("obj", "boolean"), lambda x: x["shape"] == "letter"),
  types.new_function("package", ("obj", "boolean"), lambda x: x["shape"] == "package"),

  types.new_function("male", ("obj", "boolean"), lambda x: not x["female"]),
  types.new_function("female", ("obj", "boolean"), lambda x: x["female"]),

  types.new_function("young", ("obj", "boolean"), lambda x: x["age"] < 20),
  types.new_function("old", ("obj", "boolean"), lambda x: x["age"] > 20),

  types.new_function("object", (types.ANY_TYPE, "boolean"), fn_object),
  types.new_function("agent", ("obj", "boolean"), lambda x: x["agent"]),

  # TODO types don't make sense here
  types.new_function("move", ("obj", ("obj", "boolean"), "manner", "action"), lambda obj, dest, manner: Move(obj, dest, manner)),
  types.new_function("cause_possession", ("obj", "obj", "action"), lambda agent, obj: CausePossession(agent, obj)),
  types.new_function("transfer", ("obj", "obj", "manner", "action"), lambda obj, agent, dist: Transfer(obj, agent, dist)),

  types.new_function("do_", ("action", "action", "action"), lambda a1, a2: a1 + a2),
]

constants = [types.new_constant("any", "manner"),
             types.new_constant("far", "manner"),
             types.new_constant("near", "manner"),
             types.new_constant("slow", "manner"),
             types.new_constant("fast", "manner"),
             types.new_constant("pos", "manner"),
             types.new_constant("neg", "manner")]

ontology = Ontology(types, functions, constants, variable_weight=0.1)


lexicon = Lexicon.fromstring(r"""
  :- S, N

  woman => N {\x.and_(agent(x),female(x))}
  man => N {\x.and_(agent(x),male(x))}

  letter => N {\x.letter(x)}
  ball => N {\x.sphere(x)}
  package => N {\x.package(x)}

  the => N/N {\x.unique(x)}

  give => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,any))}
  send => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,far))}
  hand => S/N/N {\a x.do_(cause_possession(a,x),transfer(x,a,near))}
  """, ontology, include_semantics=True)


scene = {
  "objects": [
    frozendict({"female": True, "agent": True, "age": 18, "shape": "person"}),
    frozendict({"shape": "letter"}),
    frozendict({"shape": "package"}),
    frozendict({"female": False, "agent": True, "age": 5, "shape": "person"}),
    frozendict({"shape": "sphere"}),
  ]
}
scene2 = {
  "objects": [
    frozendict({"female": True, "agent": True, "age": 9, "shape": "person"}),
    frozendict({"shape": "sphere"}),
  ]
}


examples = [
    ("send the woman the package", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][2]),
                    Transfer(scene["objects"][2], scene["objects"][0], "far"))),
    ("send the boy the package", scene,
     ComposedAction(CausePossession(scene["objects"][3], scene["objects"][2]),
                    Transfer(scene["objects"][2], scene["objects"][3], "far"))),
    ("hand the kid the ball", scene2,
     ComposedAction(CausePossession(scene2["objects"][0], scene2["objects"][1]),
                    Transfer(scene2["objects"][1], scene2["objects"][0], "near"))),
    ("give the lady the ball", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][4]),
                    Transfer(scene["objects"][4], scene["objects"][0], "any"))),

    ("gorp the woman the letter", scene,
     ComposedAction(CausePossession(scene["objects"][0], scene["objects"][1]),
                    Transfer(scene["objects"][1], scene["objects"][0], "far"))),
]

