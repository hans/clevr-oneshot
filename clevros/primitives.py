from frozendict import frozendict
import operator


class Event(object):

  def __init__(self):
    pass

  def __getitem__(self, attr):
    return EventOp(self, getattr, attr)

  def __call__(self):
    # Dummy method which allows us to use an instance of this class as a
    # function in the ontology.
    return None

  def __str__(self):
    return "<event>"

  __repr__ = __str__


class EventOp(object):
  """
  Lazy-evaluated operation on an event object.
  """

  def __init__(self, base, op, *args):
    self.base = base
    self.op = op
    self.args = tuple(args)

  def __hash__(self):
    return hash((self.base, self.op, self.args))

  # TODO ambiguous API semantics -- should this yield a new EventOp, or should
  # it be used to compare with another EventOp? Typing won't save us here,
  # either...
  def __eq__(self, other):
    return hash(self) == hash(other)

  def __getitem__(self, attr):
    return EventOp(self, getattr, attr)

  def __add__(self, other):
    return EventOp(self, operator.add, other)

  def __sub__(self, other):
    return EventOp(self, operator.sub, other)

  def __mul__(self, other):
    return EventOp(self, operator.mul, other)

  def __rmul__(self, other):
    return EventOp(self, operator.mul, other)

  def __lt__(self, other):
    return EventOp(self, operator.lt, other)

  def __gt__(self, other):
    return EventOp(self, operator.gt, other)

  def __str__(self, verbose=False):
    if verbose:
      op_str = repr(self.op)
    else:
      if hasattr(self.op, "__name__"):
        op_str = self.op.__name__
      elif hasattr(self.op, "__call__"):
        op_str = self.op.__class__.__name__
      else:
        op_str = str(self.op)
    return "EventOp<%s>(%s, %s)" % \
        (op_str, self.base, ", ".join(str(arg) for arg in self.args))

  def __repr__(self):
    return self.__str__(verbose=True)


def fn_unique(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]

def fn_cmp_pos(ax, manner, a, b):
  sign = 1 if manner == "pos" else -1
  return sign * (a["3d_coords"][ax()] - b["3d_coords"][ax()])

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

class Constraint(object):
  def __init__(self, *constraints):
    self.constraints = tuple(constraints)

  def __add__(self, other):
    return Constraint(self.constraints + other.constraints)

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __str__(self):
    return "Constraint(%s)" % (", ".join(map(str, self.constraints)))

  __repr__ = __str__

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
