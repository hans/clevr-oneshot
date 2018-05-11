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


class Action(object): pass
class Move(Action):
  def __init__(self, obj, dest, manner):
    self.obj = obj
    self.dest = dest
    self.manner = manner

class Transfer(Move):
  pass
