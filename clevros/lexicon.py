"""
Tools for updating and expanding lexicons, dealing with logical forms, etc.
"""

import copy

from nltk.sem.logic import *


def lf_parts(lf_str):
    """
    Parse a logical form string into a set of candidate lexical items which
    could be combined to produce the original LF.

    >>> sorted(map(str, lf_parts("filter_shape(scene,'sphere')")))
    ["'sphere'", "\\\\x.filter_shape(scene,'sphere')", '\\\\x.filter_shape(scene,x)', "\\\\x.filter_shape(x,'sphere')", 'scene']
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

        # Create the candidate node(s).
        variable = Variable("x")
        for i, arg in enumerate(node.args):
            if isinstance(arg, ApplicationExpression):
                new_arg_cands = [VariableExpression(variable)]
            else:
                new_arg_cands = [arg]
                if n_constants == len(node.args):
                    # All args are constant, so turning each constant into
                    # a variable is also legal. Do that.
                    new_arg_cands.append(VariableExpression(variable))

            # Enumerate candidate new arguments and yield candidate new exprs.
            for new_arg_cand in new_arg_cands:
                new_args = node.args[:i] + [new_arg_cand] + node.args[i + 1:]
                app_expr = ApplicationExpression(node.pred, new_args[0])
                app_expr = reduce(lambda x, y: ApplicationExpression(x, y), new_args[1:], app_expr)
                candidates.add(LambdaExpression(variable, app_expr))

    return candidates


if __name__ == '__main__':
    print(list(map(str, lf_parts("filter_shape(scene,'sphere')"))))
