"""
Tools for updating and expanding lexicons, dealing with logical forms, etc.
"""

import copy

from nltk.sem.logic import *


def lf_parts(lf_str):
    """
    Parse a logical form string into a set of candidate lexical items which
    could be combined to produce the original LF.

    >>> lf_parts("filter_shape(scene,'sphere')")
    ["'sphere'", "scene", "\\x.filter_shape(x,'sphere')", "\\x.filter_shape(scene,x)"]
    """
    # Parse into a lambda calculus expression.
    expr = Expression.fromstring(lf_str)
    assert isinstance(expr, ApplicationExpression)

    # First candidates: all available constants
    candidates = expr.constants()

    # All level-1 abstractions of the LF
    queue = [expr]
    while queue:
        node = queue.pop()

        n_constants = 0
        for arg in node.args:
            if isinstance(arg, ConstantExpression):
                n_constants += 1
            elif isinstance(arg, ApplicationExpression):
                queue.append(arg)
            else:
                assert False, "Unexpected type " + str(arg)

        # Hard constraint for now: all but one variable should be a
        # constant expression.
        if n_constants < len(node.args) - 1:
            continue

        # Create the candidate node.
        new_expr = None
        variable = Variable("x")
        for arg in node.args:
            if isinstance(arg, ApplicationExpression):
                print("hi")
                new_arg = VariableExpression(variable)
            else:
                new_arg = arg

            base = node.pred if new_expr is None else new_expr
            new_expr = ApplicationExpression(base, new_arg)

        candidates.add(LambdaExpression(variable, new_expr))

        # TODO also abstract out possible constants here

    return candidates


if __name__ == '__main__':
    print(lf_parts("filter_shape(scene,'sphere')"))
