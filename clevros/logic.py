"""
Functions for dealing with a logical language.
"""

import copy
import functools
import inspect
import itertools

from nltk.sem import logic as l


def make_application(fn_name, args):
  expr = l.ApplicationExpression(l.ConstantExpression(l.Variable(fn_name)),
                                 args[0])
  return reduce(lambda x, y: l.ApplicationExpression(x, y), args[1:], expr)


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


def extract_lambda(expr):
  """
  Extract `LambdaExpression` arguments to the top of a semantic form.
  This makes them compatible with the CCG parsing setup, which needs top-level
  lambdas in order to perform function application during parsing.
  """
  variables = []
  expr = copy.deepcopy(expr)

  def process_lambda(lambda_expr):
    # Create a new unique variable and substitute.
    unique = unique_variable()
    new_expr = lambda_expr.term.replace(lambda_expr.variable, l.IndividualVariableExpression(unique))
    return unique, new_expr

  # Traverse the LF and replace lambda expressions wherever necessary.
  def inner(node):
    if isinstance(node, l.ApplicationExpression):
      new_args = []

      for arg in node.args:
        if isinstance(arg, l.LambdaExpression):
          new_var, new_arg = process_lambda(arg)

          variables.append(new_var)
          new_args.append(new_arg)
        else:
          new_args.append(inner(arg))

      return make_application(node.pred.variable.name, new_args)
    else:
      return node

  expr = inner(expr)
  for variable in variables[::-1]:
    expr = l.LambdaExpression(variable, expr)

  return expr.normalize()


def get_callable_arity(c):
  return len(inspect.getargspec(c).args)


def as_ec_sexpr(expr):
  """
  Convert an `nltk.sem.logic` `Expression` to an S-expr string.
  """
  var_map = {}
  def inner(expr):
    if isinstance(expr, l.LambdaExpression):
      # Add lambda variable to var map.
      var_idx = len(var_map)
      var_map[expr.variable.name] = var_idx
      return "(lambda %s)" % inner(expr.term)
    elif isinstance(expr, l.ApplicationExpression):
      args = [inner(arg) for arg in expr.args]
      return "(%s %s)" % (expr.pred.variable.name, " ".join(args))
    elif isinstance(expr, l.IndividualVariableExpression):
      return "$%i" % var_map[expr.variable.name]
    elif isinstance(expr, l.ConstantExpression):
      return expr.variable.name
    else:
      raise ValueError("un-handled expression component %r" % expr)

  return inner(expr)


class Ontology(object):
  """
  TODO
  """

  def __init__(self, function_names, function_defs, function_weights, variable_weight=0.1):
    """
    Arguments:
      function_names: List of `k` function name strings
      function_defs: List of `k` Python functions
      function_weights: ndarray of dim `k`; a total ordering over the `k` functions.
      variable_weight: log-probability of observing any variable
    """
    # TODO do we need to require explicit (log-)probability distributions here?
    # I don't think so. Just need a total ordering.
    assert len(function_names) == len(function_defs)
    assert len(function_defs) == len(function_weights)
    self.function_names = function_names
    self.function_defs = function_defs
    self.function_arities = [get_callable_arity(defn) for defn in self.function_defs]
    self.function_weights = function_weights
    self.variable_weight = variable_weight

  EXPR_TYPES = [l.ApplicationExpression, l.ConstantExpression,
                l.IndividualVariableExpression, l.LambdaExpression]

  def next_bound_var(self, bound_vars):
    name_length = 1 + len(bound_vars) // 26
    name_id = len(bound_vars) % 26
    name = chr(97 + name_id)
    return Variable(name * name_length)

  def iter_expressions(self, max_depth=3):
    ret = self._iter_expressions_inner(max_depth, bound_vars=())

    # Extract lambda arguments to the top level.
    ret = [extract_lambda(expr) for expr in ret]

    return ret

  @functools.lru_cache(maxsize=None)
  @listify
  def _iter_expressions_inner(self, max_depth, bound_vars):
    if max_depth == 0:
      return

    for expr_type in self.EXPR_TYPES:
      if expr_type == l.ApplicationExpression and max_depth > 1:
        # Loop over functions according to their weights.
        fns_sorted = sorted(enumerate(self.function_names),
                            key=lambda val: self.function_weights[val[0]],
                            reverse=True)
        for idx, fn_name in fns_sorted:
          arity = self.function_arities[idx]
          sub_args = self._iter_expressions_inner(max_depth=max_depth - 1,
                                                  bound_vars=bound_vars)

          for arg_combs in itertools.product(sub_args, repeat=arity):
            candidate = make_application(fn_name, arg_combs)
            if self._valid_application_expr(candidate):
              yield candidate
      elif expr_type == l.LambdaExpression and max_depth > 1:
        bound_var = self.next_bound_var(bound_vars)

        results = self._iter_expressions_inner(max_depth=max_depth - 1,
                                               bound_vars=bound_vars + (bound_var,))
        for expr in results:
          candidate = l.LambdaExpression(bound_var, expr)
          if self._valid_lambda_expr(candidate):
            yield candidate
      elif expr_type == l.IndividualVariableExpression:
        for bound_var in bound_vars:
          yield l.IndividualVariableExpression(bound_var)

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
    while isinstance(expr, l.LambdaExpression):
      bound_args.append(expr.variable)
      expr = expr.term
    body = expr

    # Exclude exprs which do not use all of their bound arguments.
    if set(bound_args) != set(body.variables()):
      return False

    # Exclude exprs with simplistic bodies.
    if isinstance(body, l.IndividualVariableExpression):
      return False

    return True
