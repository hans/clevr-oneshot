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

  #@functools.lru_cache(maxsize=None)
  def iter_expressions(self, max_depth=3, bound_vars=()):
    if max_depth == 0:
      return

    for expr_type in self.EXPR_TYPES:
      if expr_type == ApplicationExpression and max_depth > 1:
        for arity, fns in self.functions_by_arity.items():
          for fn_name in fns:
            sub_args = self.iter_expressions(max_depth=max_depth - 1, bound_vars=bound_vars)

            for arg_combs in itertools.product(sub_args, repeat=arity):
              yield make_application(fn_name, arg_combs)
      # elif expr_type == ConstantExpression:
      #   for constant in self.constants:
      #     yield ConstantExpression(Variable(constant))
      elif expr_type == LambdaExpression and max_depth > 1:
        bound_var = self.next_bound_var(bound_vars)
        for expr in self.iter_expressions(
            max_depth=max_depth - 1,
            bound_vars=bound_vars + (bound_var,)):
          # Skip meaningless lambda bodies.
          if not isinstance(expr, IndividualVariableExpression):
            yield LambdaExpression(bound_var, expr)
      elif expr_type == IndividualVariableExpression:
        for bound_var in bound_vars:
          yield IndividualVariableExpression(bound_var)
