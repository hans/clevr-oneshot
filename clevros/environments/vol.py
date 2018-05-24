from clevros.lexicon import lexicon
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
  :- S, PP, N

  cube => N {\x.and_(object(x),cube(x))}
  sphere => N {\x.and_(object(x),sphere(x))}
  donut => N {\x.and_(object(x),donut(x))}
  hose => N {\x.and_(object(x),hose(x))}
  cylinder => N {\x.and_(object(x),cylinder(x))}
  pyramid => N {\x.and_(object(x),pyramid(x))}

  the => N/N {\x.unique(x)}

  below => PP/N {\a.constraint(ltzero(cmp_pos(ax_z,pos,e,a)))}
  right_of => PP/N {\a.constraint(ltzero(cmp_pos(ax_x,neg,e,a)))}
  in_front_of => PP/N {\a.constraint(ltzero(cmp_pos(ax_y,neg,e,a)))}

  put => S/N/PP {\b a.move(a,b,slow)}
  drop => S/N/PP {\b a.move(a,b,fast)}
  """, ontology=ontology, include_semantics=True)


scene_vol = \
  {'directions': {'above': [0.0, 0.0, 1.0],
                  'behind': [-0.754490315914154, 0.6563112735748291, 0.0],
                  'below': [-0.0, -0.0, -1.0],
                  'front': [0.754490315914154, -0.6563112735748291, -0.0],
                  'left': [-0.6563112735748291, -0.7544902563095093, 0.0],
                  'right': [0.6563112735748291, 0.7544902563095093, -0.0]},
 'image_filename': 'CLEVR_train_000002.png',
 'image_index': 2,
 'objects': [
             frozendict({
               '3d_coords': (2.1141371726989746,
                            1.0,
                            2),
              'color': 'yellow',
              'material': 'metal',
              'rotation': 308.49217566676606,
              'shape': 'sphere',
              'size': 'large'}),
            frozendict({
              '3d_coords': (0, 0, 0),
              'color': 'blue',
              'material': 'rubber',
              'pixel_coords': (188, 94, 12.699371337890625),
              'rotation': 82.51702981683107,
              'shape': 'pyramid',
              'size': 'large'}),
             frozendict({
               '3d_coords': (-2.3854215145111084,
                            0.0,
                            0.699999988079071),
              'color': 'blue',
              'material': 'rubber',
              'rotation': 82.51702981683107,
              'shape': 'cube',
              'size': 'large'})],
 'split': 'train'}


examples = [
  ("place the sphere right_of the cube", scene_vol,
   Move(scene_vol["objects"][0],
        Constraint(-1 * (Event()["3d_coords"][0] - scene_vol["objects"][2]["3d_coords"][0]) < 0),
        "slow")),
]

