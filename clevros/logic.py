"""
Functions for dealing with a logical language.
"""

import functools
import inspect
import itertools

from nltk.sem.logic import *


def make_application(fn_name, args):
  expr = ApplicationExpression(ConstantExpression(Variable(fn_name)),
                               args[0])
  return reduce(lambda x, y: ApplicationExpression(x, y), args[1:], expr)


def listify(fn=None, wrapper=list):
  """
  A decorator which wraps a function's return value in ``list(...)``.

  Useful when an algorithm can be expressed more cleanly as a generator but
  the function should return an list.
  """
  def listify_return(fn):
    @functools.wraps(fn)
    def listify_helper(*args, **kw):
      return wrapper(fn(*args, **kw))
    return listify_helper
  if fn is None:
    return listify_return
  return listify_return(fn)


class Ontology(object):
  """
  TODO
  """

  def __init__(self, functions):
    self.functions = functions
    self.functions_by_arity = {
        count: set(fns)
        for count, fns in itertools.groupby(self.functions.keys(),
          lambda fn_name: len(inspect.getargspec(self.functions[fn_name]).args))}

    self.constants = ["sphere"] # TODO

    print(self.functions_by_arity)
    self.max_arity = max(self.functions_by_arity.keys())

  EXPR_TYPES = [ApplicationExpression, ConstantExpression, IndividualVariableExpression, LambdaExpression]

  def next_bound_var(self, bound_vars):
    name_length = 1 + len(bound_vars) // 26
    name_id = len(bound_vars) % 26
    name = chr(97 + name_id)
    return Variable(name * name_length)

  @functools.lru_cache(maxsize=None)
  @listify
  def iter_expressions(self, max_depth=3, bound_vars=()):
    if max_depth == 0:
      return

    for expr_type in self.EXPR_TYPES:
      if expr_type == ApplicationExpression and max_depth > 1:
        for arity, fns in self.functions_by_arity.items():
          for fn_name in fns:
            sub_args = self.iter_expressions(max_depth=max_depth - 1,
                                             bound_vars=bound_vars)

            for arg_combs in itertools.product(sub_args, repeat=arity):
              candidate = make_application(fn_name, arg_combs)
              if self._valid_application_expr(application_expr):
                yield application_expr
      elif expr_type == LambdaExpression and max_depth > 1:
        bound_var = self.next_bound_var(bound_vars)

        results = self.iter_expressions(max_depth=max_depth - 1,
                                        bound_vars=bound_vars + (bound_var,))
        for expr in results:
          candidate = LambdaExpression(bound_var, expr)
          if self._valid_lambda_expr(candidate):
            yield candidate
      elif expr_type == IndividualVariableExpression:
        for bound_var in bound_vars:
          yield IndividualVariableExpression(bound_var)

  def _valid_application_expr(self, application_expr):
    """
    Check whether this `ApplicationExpression` should be considered when
    enumerating programs.
    """
    # TODO check type consistency
    return True

  def _valid_lambda_expr(self, lambda_expr):
    """
    Check whether this `LambdaExpression` should be considered when enumerating
    programs.
    """

    # Collect bound arguments and the body expression.
    bound_args = []
    expr = lambda_expr
    while isinstance(expr, LambdaExpression):
      bound_args.append(expr.variable)
      expr = expr.term
    body = expr

    # Exclude exprs which do not use all of their bound arguments.
    if set(bound_args) != set(body.variables()):
      return False

    # Exclude exprs with simplistic bodies.
    if isinstance(body, IndividualVariableExpression):
      return False

    return True
