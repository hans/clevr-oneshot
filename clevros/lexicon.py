"""
Tools for updating and expanding lexicons, dealing with logical forms, etc.
"""

import copy

from nltk.ccg import lexicon
from nltk.ccg.api import PrimitiveCategory
from nltk.sem.logic import *

from clevros.chart import WeightedCCGChartParser
from clevros.clevr import scene_candidate_referents


def token_categories(lex):
  """
  Return a set of categories which a new token can take on.
  """
  return set([lexicon.augParseCategory(prim, lex._primitives,
                                       lex._families)[0]
              for prim in lex._primitives])


def augment_lexicon(old_lex, sentence, lf):
  """
  Augment an existing lexicon to cover the elements present in a new
  sentence--logical form pair.

  This is the first step of the standard "GENLEX" routine.
  Will create an explosion of possible word-meaning pairs. Many of these
  form-meaning pairs won't be valid -- i.e., they won't combine with other
  elements of the lexicon in any way to yield the original parse `lf`.

  Args:
    lexicon: CCGLexicon
    sentence: list of string tokens
    lf: LF string
  Returns:
    A modified deep copy of `lexicon`.
  """

  new_lex = copy.deepcopy(old_lex)

  lf_cands = lf_parts(lf)
  cat_cands = token_categories(old_lex)
  for word in sentence:
    if not new_lex.categories(word):
      for category in cat_cands:
        for lf_cand in lf_cands:
          if (isinstance(lf_cand, ConstantExpression)
            and not isinstance(category, PrimitiveCategory)):
            # Syntactic type does not match semantic type. Skip.
            # TODO: Handle more than the dichotomy between
            # constant and argument-taking expressions (e.g.
            # multi-argument lambdas).
            continue

          new_token = lexicon.Token(word, category, lf_cand, 1.0)
          new_lex._entries[word].append(new_token)

  return new_lex


def augment_lexicon_scene(old_lex, sentence, scene):
  """
  Augment a lexicon to cover the words in a new sentence uttered in some
  scene context.

  Args:
    old_lex: CCGLexicon
    sentence: list of word tokens
    scene: CLEVR scene
  """

  lex = copy.deepcopy(old_lex)

  # TODO(long term): first run a parse without semantics and see which
  # syntactic categories allow new words to yield valid parses (and maybe
  # also answer questions correctly). Use these type restrictions to
  # constrain the lexicon augmentation.

  # For now, every new word receives a similarly enormous explosion of LF
  # candidates.

  lf_cands = scene_candidate_referents(scene)
  cat_cands = token_categories(lex)

  for word in sentence:
    if not lex.categories(word):
      for category in cat_cands:
        for lf_cand in lf_cands:
          new_token = lexicon.Token(word, category, lf_cand, 1.0)
          lex._entries[word].append(new_token)

  return lex


def filter_lexicon_entry(lexicon, entry, sentence, lf):
  """
  Filter possible syntactic/semantic mappings for a given lexicon entry s.t.
  the given sentence renders the given LF, holding everything else
  constant.

  This process is of course not fail-safe -- the rest of the lexicon must
  provide the necessary definitions to guarantee that any valid parse can
  result.

  Args:
    lexicon: CCGLexicon
    entry: string word
    sentence: list of word tokens, must contain `entry`
    lf: logical form string
  """
  if entry not in sentence:
    raise ValueError("Sentence does not contain given entry")

  entry_idxs = [i for i, val in enumerate(sentence) if val == entry]
  parse_results = WeightedCCGChartParser(lexicon).parse(sentence, True)

  valid_cands = [set() for _ in entry_idxs]
  for _, _, edge_cands in parse_results:
    for entry_idx, valid_cands_set in zip(entry_idxs, valid_cands):
      valid_cands_set.add(edge_cands[entry_idx])

  # Find valid interpretations across all uses of the word in the
  # sentence.
  valid_cands = list(reduce(lambda x, y: x & y, valid_cands))
  if not valid_cands:
    raise ValueError("no consistent interpretations of word found.")

  new_lex = copy.deepcopy(lexicon)
  new_lex._entries[entry] = [cand.token() for cand in valid_cands]

  return new_lex


def lf_parts(lf_str):
  """
  Parse a logical form string into a set of candidate lexical items which
  could be combined to produce the original LF.

  >>> sorted(map(str, lf_parts("filter_shape(scene,'sphere')")))
  ["'sphere'", "\\\\x.filter_shape(scene,'sphere')", '\\\\x.filter_shape(scene,x)', "\\\\x.filter_shape(x,'sphere')", 'scene']
  """
  # TODO avoid producing lambda expressions which don't make use of
  # their arguments.

  # Parse into a lambda calculus expression.
  expr = Expression.fromstring(lf_str)
  assert isinstance(expr, ApplicationExpression)

  # First candidates: all available constants
  candidates = set([ConstantExpression(const)
            for const in expr.constants()])

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
  print(list(map(str, lf_parts("filter_shape(filter_size(scene, 'big'), 'sphere')"))))

  lex = lexicon.fromstring(r"""
  :- NN, DET, ADJ

  DET :: NN/NN
  ADJ :: NN/NN

  the => DET {\x.unique(x)}
  big => ADJ {\x.filter_size(x,big)}
  dog => NN {dog}""", include_semantics=True)
  print(augment_lexicon(lex, "the small dog".split(), "unique(filter_size(dog,small))"))
