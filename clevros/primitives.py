from frozendict import frozendict


def fn_unique(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]

def fn_cmp_pos(ax, a, b): return a["3d_coords"][ax()] - b["3d_coords"][ax()]
def fn_ltzero(x): return x < 0
def fn_and(a, b): return a and b

def fn_ax_x(): return 0
def fn_ax_y(): return 1
def fn_ax_z(): return 2

def fn_cube(x): return x["shape"] == "cube"
def fn_sphere(x): return x["shape"] == "sphere"
def fn_donut(x): return x["shape"] == "donut"
def fn_pyramid(x): return x["shape"] == "pyramid"
def fn_hose(x): return x["shape"] == "hose"
def fn_cylinder(x): return x["shape"] == "cylinder"

def fn_object(x): return isinstance(x, (frozendict, dict))


class Action(object):
  def __add__(self, other):
    return ComposedAction(self, other)

  def __eq__(self, other):
    return hash(self) == hash(other)

class ComposedAction(Action):
  def __init__(self, *actions):
    self.actions = actions

  def __hash__(self):
    return hash(tuple(self.actions))

  def __str__(self):
    return "+(%s)" % (",".join(str(action) for action in self.actions))

  __repr__ = __str__

class Move(Action):
  def __init__(self, obj, dest, manner):
    self.obj = obj
    self.dest = dest
    self.manner = manner

  def __hash__(self):
    return hash((self.obj, self.dest, self.manner))

  def __str__(self):
    return "%s(%s -> %s, %s)" % (self.__class__.__name__, self.obj, self.dest, self.manner)

  __repr__ = __str__

class Transfer(Move):
  pass

class StateChange(Action): pass
class CausePossession(StateChange):
  def __init__(self, agent, obj):
    self.agent = agent
    self.obj = obj

  def __hash__(self):
    return hash((self.agent, self.obj))

  def __str__(self):
    return "%s(%s <- %s)" % (self.__class__.__name__, self.agent, self.obj)

  __repr__ = __str__
