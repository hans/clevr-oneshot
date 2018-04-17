"""
Model for evaluation of logical forms on CLEVR-like scenes.
"""

from nltk.sem.logic import *


class Model(object):
  """
  Grounded logical evaluation model, mostly stolen from `nltk.sem.evaluate`.
  """

  def __init__(self, valuation, functions=None):
    self.valuation = valuation
    self.functions = functions or []

  def evaluate(self, expr, scene):
    return self.satisfy(expr, scene)

  def satisfy(self, expr, scene):
    """
    Recursively interpret an expression in the context of some scene.
    """
    print(expr)
    if isinstance(expr, ApplicationExpression):
      function, arguments = expr.uncurry()
      if isinstance(function, AbstractVariableExpression):
        #It's a predicate expression ("P(x,y)"), so used uncurried arguments
        funval = self.satisfy(function, scene)
        argvals = tuple(self.satisfy(arg, scene) for arg in arguments)

        if callable(funval):
          return funval(*argvals)
        return argvals in funval
      else:
        #It must be a lambda expression, so use curried form
        funval = self.satisfy(expr.function, scene)
        argval = self.satisfy(expr.argument, scene)
        return funval[argval]
    elif isinstance(expr, NegatedExpression):
      return not self.satisfy(expr.term, scene)
    elif isinstance(expr, AndExpression):
      return self.satisfy(expr.first, scene) and \
              self.satisfy(expr.second, scene)
    elif isinstance(expr, OrExpression):
      return self.satisfy(expr.first, scene) or \
              self.satisfy(expr.second, scene)
    elif isinstance(expr, ImpExpression):
      return (not self.satisfy(expr.first, scene)) or \
              self.satisfy(expr.second, scene)
    elif isinstance(expr, IffExpression):
      return self.satisfy(expr.first, scene) == \
              self.satisfy(expr.second, scene)
    elif isinstance(expr, EqualityExpression):
      return self.satisfy(expr.first, scene) == \
              self.satisfy(expr.second, scene)
    elif isinstance(expr, AllExpression):
      new_g = g.copy()
      for u in self.domain:
        new_g.add(expr.variable.name, u)
        if not self.satisfy(expr.term, new_scene):
          return False
      return True
    elif isinstance(expr, ExistsExpression):
      new_g = g.copy()
      for u in self.domain:
        new_g.add(expr.variable.name, u)
        if self.satisfy(expr.term, new_scene):
          return True
      return False
    elif isinstance(expr, LambdaExpression):
      cf = {}
      var = expr.variable.name
      for u in self.domain:
        val = self.satisfy(expr.term, g.add(var, u))
        # NB the dict would be a lot smaller if we do this:
        # if val: cf[u] = val
        # But then need to deal with cases where f(a) should yield
        # a function rather than just False.
        cf[u] = val
      return cf
    else:
      return self.i(expr, scene)

  def i(self, expr, scene):
    """
    An interpretation function.

    Assuming that ``expr`` is atomic:

    - if ``expr`` is a non-logical constant, calls the valuation *V*
    - else if ``expr`` is an individual variable, calls assignment *g*
    - else returns ``Undefined``.

    :param expr: an ``Expression`` of ``logic``.
    :type g: Assignment
    :param g: an assignment to individual variables.
    :return: a semantic value
    """
    # If expr is a propositional letter 'p', 'q', etc, it could be in valuation.symbols
    # and also be an IndividualVariableExpression. We want to catch this first case.
    # So there is a procedural consequence to the ordering of clauses here:
    if expr.variable.name in self.functions:
      return self.functions[expr.variable.name]
    elif expr.variable.name in self.valuation:
      valuation = self.valuation[expr.variable.name]
      if callable(valuation):
        return valuation(scene)
      return valuation
    elif isinstance(expr, IndividualVariableExpression):
      return g[expr.variable.name]
    else:
      raise Undefined("Can't find a value for %s" % expr)
