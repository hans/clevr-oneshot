from nltk.ccg.api import FunctionalCategory
from nltk.ccg.combinator import DirectedBinaryCombinator
from nltk.sem import logic as l


def make_unique_variable(expr, var_class=None, allow_clash=None):
  """
  Generate a unique variable after Variable `pattern` in the context of some
  `expr.`
  """
  disallow_clash = set(expr.free()) - set(allow_clash or [])

  prefixes = {
    "function": "F",
    "individual": "z",
    None: "z"
  }
  prefix = prefixes[var_class]

  i = 1
  while True:
    var = l.Variable("%s%s" % (prefix, i))
    if not var in disallow_clash:
      break
    i += 1
  return var


class PositionalForwardRaiseCombinator(DirectedBinaryCombinator):
  r"""
  CCG type operation which type-raises a child category in arbitrary position
  and adds the type-raising arguments as arguments in the forward direction to
  the parent category.

  NB: Only functional categories with forward directions are currently
  supported.

  Not always a valid operation -- requires that the child category's semantics
  be some higher-order function. Deploy with care.

      X/Y           Z   ==>PFR(0)  X/(Y/Z)/Z
      \y. x(y)                     \z.\Y. x(y(z))

      W/X/Y         Z   ==>PFR(0)  W/X/(Y/Z)/Z
      \y x.w(x, y)                 \z.\Y x.w(x,y(z))

      W/X/Y         Z   ==>PFR(1)  W/(X/Z)/Y/Z
      \y x.w(x, y)                 \z.\y X.w(x(z),y)

  With index 0, this operation yields a sort of intermediate step of a
  composition operation. Consider the following composition:

      X/Y  Y/Z   ==>   X/Z

  an intermediate step of the above would look like (before simplification)

      X/(Y/Z)/Z

  and this is exactly what this combinator (as `PFR(0)`) would yield.
  """

  def __init__(self, index=0):
    """
    Construct a PFR operation acting `index` steps from the rightmost forward
    argument of the left category. See class documentation for examples of how
    the index argument functions.
    """
    if index != 0:
      raise NotImplementedError("PFR with index > 0 not currently supported.")
    self.index = index

  def can_combine(self, left, right):
    if not (left.is_function() and right.is_primitive()):
      return False

    # Verify that index applies here.
    arity = 0
    while arity < self.index:
      if not left.is_function():
        return False
      elif not left.dir().is_forward():
        raise NotImplementedError("Only forward application is currently supported.")

      left = left.arg()
      arity += 1

    return True

  def combine(self, left, right):
    # Below implementation is specific for `index == 0`.

    # Type-raise the argument at index 0.
    raised_arg = FunctionalCategory(left.arg(), right, left.dir())
    left = FunctionalCategory(left.res(), raised_arg, left.dir())
    return FunctionalCategory(left, right, left.dir())

  def update_semantics(self, semantics):
    parent = None
    node, index = semantics, self.index
    while index > 0:
      if not isinstance(node, l.LambdaExpression):
        raise ValueError("semantics provided do not have sufficient arity to extract argument at position %i" % self.index)

      parent = node
      node = node.term

    # Convert target variable to a uniquely named function variable.
    target_variable = node.variable
    new_target_var = make_unique_variable(node.term, "function")

    # Create an application expression applying the raised target var to some
    # extracted argument.
    extracted_arg = make_unique_variable(semantics, allow_clash=[target_variable])
    application_expr = l.ApplicationExpression(l.FunctionVariableExpression(new_target_var),
                                               l.IndividualVariableExpression(extracted_arg))

    node.term = node.term.replace(target_variable, application_expr)
    node.variable = new_target_var

    if parent is not None:
      parent.term = node

    return l.LambdaExpression(extracted_arg, semantics)
