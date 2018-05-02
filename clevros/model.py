"""
Model for evaluation of logical forms on CLEVR-like scenes.
"""

from copy import deepcopy

from nltk.sem.logic import *


class Model(object):
  """
  Grounded logical evaluation model, mostly stolen from `nltk.sem.evaluate`.
  """

  def __init__(self, scene, ontology):
    self.scene = scene
    self.ontology = ontology
    self.domain = deepcopy(scene["objects"])

  def evaluate(self, expr):
    return self.satisfy(expr)

  def satisfy(self, expr, assignments=None):
    """
    Recursively interpret an expression in the context of some scene.
    """
    if assignments is None:
      assignments = {}

    if isinstance(expr, ApplicationExpression):
      function, arguments = expr.uncurry()
      if isinstance(function, AbstractVariableExpression):
        #It's a predicate expression ("P(x,y)"), so used uncurried arguments
        funval = self.satisfy(function, assignments)

        if isinstance(funval, LambdaExpression):
          # Function is defined in terms of other functions. Do beta reduction
          # to get a proper program defined in terms of low-level functions.
          program = funval(*arguments).simplify()

          # The program can be arbitrarily complex, so we can't evaluate it
          # here as normal -- need to recurse.
          return self.satisfy(program, assignments)

        # OK, if we're still here we just have a basic low-level function.
        # Evaluate the arguments and apply.
        argvals = tuple(self.satisfy(arg, assignments) for arg in arguments)

        if callable(funval):
          return funval(*argvals)
        return argvals in funval
      else:
        #It must be a lambda expression, so use curried form
        funval = self.satisfy(expr.function, assignments)
        argval = self.satisfy(expr.argument, assignments)
        return funval[argval]
    elif isinstance(expr, NegatedExpression):
      return not self.satisfy(expr.term, assignments)
    elif isinstance(expr, AndExpression):
      return self.satisfy(expr.first, assignments) and \
              self.satisfy(expr.second, assignments)
    elif isinstance(expr, OrExpression):
      return self.satisfy(expr.first) or \
              self.satisfy(expr.second, assignments)
    elif isinstance(expr, ImpExpression):
      return (not self.satisfy(expr.first)) or \
              self.satisfy(expr.second, assignments)
    elif isinstance(expr, IffExpression):
      return self.satisfy(expr.first) == \
              self.satisfy(expr.second, assignments)
    elif isinstance(expr, EqualityExpression):
      return self.satisfy(expr.first) == \
              self.satisfy(expr.second, assignments)
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
        assignments = deepcopy(assignments)
        assignments[var] = u

        val = self.satisfy(expr.term, assignments)
        # NB the dict would be a lot smaller if we do this:
        # if val: cf[u] = val
        # But then need to deal with cases where f(a) should yield
        # a function rather than just False.
        cf[u] = val
      return cf
    else:
      return self.i(expr, assignments)

  def i(self, expr, assignments):
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
    if expr.variable.name in self.ontology.functions_dict:
      return self.ontology.functions_dict[expr.variable.name].defn
    elif isinstance(expr, IndividualVariableExpression):
      return assignments[expr.variable.name]
    else:
      print("expr:", expr)
      raise ValueError("Can't find a value for %s" % expr)
