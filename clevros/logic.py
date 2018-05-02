# -*- coding: utf-8
"""
Functions for dealing with a logical language.
"""

import copy
import functools
import inspect
import itertools
import re

from nltk.sem import logic as l


class TypeSystem(object):

  def __init__(self, primitive_types):
    self._types = {primitive_type_name: l.BasicType(l.ENTITY_TYPE)
                   for primitive_type_name in primitive_types}

  def __getitem__(self, type_expr):
    if isinstance(type_expr, l.BasicType):
      return type_expr
    if isinstance(type_expr, str):
      return self._types[type_expr]
    return self.make_function_type(type_expr)

  def __iter__(self):
    return iter(self._types.values())

  def make_function_type(self, type_expr):
    ret = self[type_expr[-1]]
    for type_expr_i in type_expr[:-1][::-1]:
      ret = l.ComplexType(self[type_expr_i], ret)
    return ret

  def new_function(self, name, type, defn, **kwargs):
    type = self[type]
    return Function(name, type, defn, **kwargs)


# Wrapper for a typed function.
class Function(object):
  """
  Wrapper for a typed function.
  """

  def __init__(self, name, type, defn, weight=0.0):
    self.name = name
    self.type = type
    self.defn = defn
    self.weight = weight

    # We can't statically verify the type of the definition, but we can at
    # least verify the arity.
    assert self.arity == get_callable_arity(self.defn)

  @property
  def arity(self):
    return len(self.type.flat) - 1

  @property
  def arg_types(self):
    return self.type.flat[:-1]

  @property
  def return_type(self):
    return self.type.flat[-1]

  def __str__(self):
    return "function %s : %s" % (self.name, self.type)


def make_application(fn_name, args):
  expr = l.ApplicationExpression(l.ConstantExpression(l.Variable(fn_name)),
                                 args[0])
  return functools.reduce(lambda x, y: l.ApplicationExpression(x, y),
                          args[1:], expr)


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
    unique = l.unique_variable()
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


def read_ec_sexpr(sexpr):
  tokens = re.split(r"([()\s])", sexpr)

  bound_vars = {}
  is_call = False
  stack = [(None, None, [])]
  for token in tokens:
    token = token.strip()
    if not token:
      continue

    if token == "(":
      is_call = True
    elif token == "lambda":
      is_call = False
      variable = next_bound_var(bound_vars, l.ANY_TYPE)
      var_idx = "$%i" % len(bound_vars)
      bound_vars[var_idx] = variable

      stack.append((l.LambdaExpression, variable, []))
    elif is_call:
      stack.append((l.ApplicationExpression, token, []))
      is_call = False
    elif token == ")":
      stack_top = stack.pop()
      if stack_top[0] == l.LambdaExpression:
        _, variable, term = stack_top
        result = l.LambdaExpression(variable, term[0])
      elif stack_top[0] == l.ApplicationExpression:
        _, pred, args = stack_top
        result = make_application(pred, args)
      else:
        raise RuntimeError("unknown element on stack", stack_top)

      stack[-1][2].append(result)
    elif token in bound_vars:
      stack[-1][2].append(l.IndividualVariableExpression(bound_vars[token]))
    else:
      stack[-1][2].append(l.ConstantExpression(l.Variable(token)))

  assert len(stack) == 1
  assert len(stack[0][2]) == 1
  return stack[0][2][0]


class Ontology(object):
  """
  TODO
  """

  def __init__(self, types, functions, variable_weight=0.1):
    """
    Arguments:
      types: TypeSystem
      functions: List of `k` `Function` instances
      variable_weight: log-probability of observing any variable
    """
    # TODO do we need to require weights as explicit (log-)probability
    # distributions here?
    # I don't think so. Just need a total ordering.

    self.types = types

    self.functions = functions
    self.functions_dict = {fn.name: fn for fn in self.functions}
    self.variable_weight = variable_weight

    self._prepare()

  EXPR_TYPES = [l.ApplicationExpression, l.ConstantExpression,
                l.IndividualVariableExpression, l.LambdaExpression]

  def _prepare(self):
    self._nltk_type_signature = self._make_nltk_type_signature()

  def iter_expressions(self, max_depth=3):
    ret = self._iter_expressions_inner(max_depth, bound_vars=())

    # Extract lambda arguments to the top level.
    # NB, this breaks the type record. Should be fine.
    ret = [extract_lambda(expr) for expr in ret]

    return ret

  #@functools.lru_cache(maxsize=None)
  @listify
  def _iter_expressions_inner(self, max_depth, bound_vars,
                              type_request=None):
    """
    Enumerate all legal expressions.

    Arguments:
      max_depth: Maximum tree depth to traverse.
      bound_vars: Bound variables (and their types) in the parent context. The
        returned expressions may reference these variables. List of `(name,
        type)` tuples.
      type_request: Optional requested type of the expression. This helps
        greatly restrict the space of enumerations when the type system is
        strong.
    """
    if max_depth == 0:
      return

    for expr_type in self.EXPR_TYPES:
      if expr_type == l.ApplicationExpression and max_depth > 1:
        # Loop over functions according to their weights.
        fns_sorted = sorted(self.functions_dict.values(),
                            key=lambda fn: fn.weight,
                            reverse=True)

        for fn in fns_sorted:
          # If there is a present type request, only consider functions with
          # the correct return type.
          # print("\t" * (6 - max_depth), fn.name, fn.return_type, " // request: ", type_request, bound_vars)
          if type_request is not None and fn.return_type != type_request:
            continue

          if fn.arity == 0:
            # 0-arity functions are represented in the logic as
            # `ConstantExpression`s.
            # print("\t" * (6 - max_depth + 1), "yielding const ", fn.name)
            yield l.ConstantExpression(l.Variable(fn.name))
          else:
            # print("\t" * (6 - max_depth), fn, fn.arg_types)
            sub_args = []
            for i, arg_type_request in enumerate(fn.arg_types):
              # print("\t" * (6 - max_depth + 1), "ARGUMENT %i (max_depth %i)" % (i, max_depth - 1))
              sub_args.append(
                  self._iter_expressions_inner(max_depth=max_depth - 1,
                                               bound_vars=bound_vars,
                                               type_request=arg_type_request))

            for arg_combs in itertools.product(*sub_args):
              candidate = make_application(fn.name, arg_combs)
              valid = self._valid_application_expr(candidate)
              # print("\t" * (6 - max_depth + 1), "valid %s? %s" % (candidate, valid))
              if self._valid_application_expr(candidate):
                yield candidate
      elif expr_type == l.LambdaExpression and max_depth > 1:
        for bound_var_type in self.types:
          bound_var = self.next_bound_var(bound_vars, bound_var_type)
          subexpr_bound_vars = bound_vars + (bound_var,)

          subexpr_type_requests = []
          if type_request is None:
            subexpr_type_requests = [None]
          else:
            # Build new type requests using the flat structure.
            type_request_flat = type_request.flat

            subexpr_type_requests.append(type_request_flat)

            # Basic case: the variable is used as one of the existing
            # arguments of a target subexpression.
            subexpr_type_requests.extend([type_request_flat[:i] + type_request_flat[i + 1:]
                                          for i, type_i in enumerate(type_request_flat)
                                          if type_i == bound_var_type])

            # # The subexpression might take this variable as an additional
            # # argument in any position. We have to enumerate all possibilities
            # # -- yikes!
            # for insertion_point in range(len(type_request)):
            #   subexpr_type_requests.append(type_request[:insertion_point] + (bound_var_type,)
            #       + type_request[insertion_point:])
          # print("\t" * (6-max_depth), "λ %s :: %s" % (bound_var, bound_var_type), type_request, subexpr_type_requests)
          # print("\t" * (6-max_depth), "Now recursing with max_depth=%i" % (max_depth - 1))

          for subexpr_type_request in subexpr_type_requests:
            if not subexpr_type_request:
              continue

            results = self._iter_expressions_inner(max_depth=max_depth - 1,
                                                    bound_vars=subexpr_bound_vars,
                                                    type_request=self.types.make_function_type(subexpr_type_request))
            for expr in results:
              candidate = l.LambdaExpression(bound_var, expr)
              valid = self._valid_lambda_expr(candidate, bound_vars)
              # print("\t" * (6 - max_depth), "valid lambda %s? %s" % (candidate, valid))
              if self._valid_lambda_expr(candidate, bound_vars):
                # Assign variable types before returning.
                typecheck_signature = copy.copy(self._nltk_type_signature)
                typecheck_signature.update({bound_var.name: bound_var.type
                                            for bound_var in subexpr_bound_vars})

                try:
                  candidate.typecheck(typecheck_signature)
                except l.InconsistentTypeHierarchyException:
                  pass
                else:
                  yield candidate
      elif expr_type == l.IndividualVariableExpression:
        for bound_var in bound_vars:
          if type_request and not bound_var.type.matches(type_request):
            continue

          # print("\t" * (6-max_depth), "var %s" % bound_var)

          yield l.IndividualVariableExpression(bound_var)

  def next_bound_var(self, bound_vars, type):
    """
    Helper function: generate the next bound variable in a context where there
    are currently the bound variables `bound_vars` (assumed to be generated by
    this function).
    """
    name_length = 1 + len(bound_vars) // 26
    name_id = len(bound_vars) % 26
    name = chr(97 + name_id)
    return l.Variable(name * name_length, type)

  def _valid_application_expr(self, application_expr):
    """
    Check whether this `ApplicationExpression` should be considered when
    enumerating programs.
    """
    # TODO check type consistency
    return True

  def _valid_lambda_expr(self, lambda_expr, ctx_bound_vars):
    """
    Check whether this `LambdaExpression` should be considered when enumerating
    programs.

    Arguments:
      lambda_expr: `LambdaExpression`
      ctx_bound_vars: Bound variables from the containing context
    """

    # TODO fails on \a b.ltzero(cmp_pos(ax_x,a,b))

    # Collect bound arguments and the body expression.
    bound_args = []
    expr = lambda_expr
    while isinstance(expr, l.LambdaExpression):
      bound_args.append(expr.variable)
      expr = expr.term
    body = expr

    # Exclude exprs which do not use all of their bound arguments.
    available_vars = set(bound_args) | set(ctx_bound_vars)
    if available_vars != set(body.variables()):
      return False

    # # Exclude exprs with simplistic bodies.
    # if isinstance(body, l.IndividualVariableExpression):
    #   return False

    return True

  def _make_nltk_type_expr(self, type_expr):
    if isinstance(type_expr, tuple) and len(type_expr) == 1:
      type_expr = type_expr[0]

    if type_expr in self.nltk_types:
      return self.nltk_types[type_expr]
    elif len(type_expr) > 1:
      return l.ComplexType(self._make_nltk_type_expr(type_expr[0]),
                           self._make_nltk_type_expr(type_expr[1:]))
    else:
      raise RuntimeError("unknown basic type %s" % (type_expr,))

  def _make_nltk_type_signature(self):
    return {fn.name: fn.type for fn in self.functions}
