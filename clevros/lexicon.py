"""
Tools for updating and expanding lexicons, dealing with logical forms, etc.
"""

from collections import defaultdict
import copy
import itertools

from nltk.ccg import lexicon as ccg_lexicon
from nltk.ccg.api import PrimitiveCategory
from nltk.sem.logic import *

from clevros import chart
from clevros.clevr import scene_candidate_referents


class Lexicon(ccg_lexicon.CCGLexicon):

  @classmethod
  def fromstring(cls, lex_str, include_semantics=False):
    return ccg_lexicon.fromstring(lex_str,
                                  include_semantics=include_semantics,
                                  cls=cls)

  def clone(self, retain_semantics=True):
    """
    Return a clone of the current lexicon instance.
    """
    ret = copy.deepcopy(self)

    if not retain_semantics:
      for entry_tokens in ret._entries.values():
        for token in entry_tokens:
          token._semantics = None

    return ret

  @property
  def primitive_categories(self):
    return set([ccg_lexicon.augParseCategory(prim, self._primitives,
                                             self._families)[0]
                for prim in self._primitives])

  @property
  def observed_categories(self):
    """
    Find categories (both primitive and functional) attested in the lexicon.
    """
    return set([token.categ()
                for token_list in self._entries.values()
                for token in token_list])


class Token(ccg_lexicon.Token):

  def __str__(self):
    return "Token(%s => %s%s)" % (self._token, self._categ,
                                  " {%s}" % self._semantics if self._semantics else "")

  __repr__ = __str__


def get_category_arity(cat):
  """
  Get the syntactic arity of a syntactic category.
  """
  if isinstance(cat, PrimitiveCategory):
    return 0
  else:
    return 1 + get_category_arity(cat.arg()) \
        + get_category_arity(cat.res())


def is_compatible(category, lf):
  """
  Determine if a syntactic category and a logical form are functionally
  compatible. (They should have the same number of arguments, unless the
  syntactic category is a simple category.)
  """
  # Get category arity by DFS.
  category_arity = get_category_arity(category)

  def visit_node(node):
    delta = 1 if isinstance(node, LambdaExpression) else 0

    try:
      res = node.visit(visit_node, sum)
    except NotImplementedError:
      res = 0
    return delta + res

  lf_arity = visit_node(lf)

  return category_arity == 0 or category_arity == lf_arity


def get_candidate_categories(lex, tokens, sentence):
  """
  Find candidate categories for the given tokens which appear in `sentence` such
  that `sentence` yields a parse.
  """
  assert set(tokens).issubset(set(sentence))

  # Make a minimal copy of `lex` which does not track semantics.
  lex = lex.clone(retain_semantics=False)

  # Remove entries for the queried tokens.
  for token in tokens:
    lex._entries[token] = []

  candidate_categories = lex.observed_categories
  ret = defaultdict(set)

  # NB does not cover the case where a single token needs multiple syntactic
  # interpretations for the sentence to parse
  for cat_assignment in itertools.product(candidate_categories, repeat=len(tokens)):
    for token, category in zip(tokens, cat_assignment):
      lex._entries[token] = [Token(token, category)]

    # Attempt a parse.
    results = chart.WeightedCCGChartParser(lex).parse(sentence)
    if results:
      # Good, we have a parse. Add candidate categories to return.
      for token, category in zip(tokens, cat_assignment):
        ret[token].add(category)

  return dict(ret)


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

  new_lex = old_lex.clone()

  lf_cands = lf_parts(lf)
  for word in sentence:
    if not new_lex.categories(word):
      for category in old_lex.primitive_categories:
        for lf_cand in lf_cands:
          if not is_compatible(category, lf_cand):
            # Arities of syntactic form and semantic form do not match.
            continue

          new_token = Token(word, category, lf_cand, 1.0)
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

  # Build a "minimal" clone of the given lexicon which tracks only
  # syntax. This minimal lexicon will be used to impose syntactic
  # constraints and prune the candidates for new words.
  old_lex_minimal = old_lex.clone(retain_semantics=False)
  minimal_parser = chart.WeightedCCGChartParser(old_lex_minimal)

  # Target lexicon to be returned.
  lex = old_lex.clone()

  lf_cands = scene_candidate_referents(scene)

  for word in sentence:
    if not lex.categories(word):
      for category in lex.primitive_categories:
        # Run a test syntactic parse to determine whether this word can
        # plausibly have this syntactic category under the grammar rules.
        #
        # TODO: It could be the case that a single word appears multiple times
        # in a sentence and has different syntactic interpretations among
        # those instances. This pruning would fail, causing that
        # sentence to have zero valid interpretations.
        minimal_token = Token(word, category)
        old_lex_minimal._entries[word].append(minimal_token)
        results = minimal_parser.parse(sentence)
        if not results:
          # Syntactically invalid candidate.
          continue

        for lf_cand in lf_cands:
          if not is_compatible(category, lf_cand):
            continue
          new_token = Token(word, category, lf_cand, 1.0)
          lex._entries[word].append(new_token)

  return lex


def augment_lexicon_distant(old_lex, query_tokens, query_token_syntaxes,
                            sentence, ontology, model, answer):
  """
  Augment a lexicon with candidate meanings for a given word using distant
  supervision. (The induced meanings for the queried words must yield parses
  that lead to `answer` under the `model`.)

  Arguments:
    old_lex: Existing lexicon which needs to be augmented. Do not write
      in-place.
    query_words: Set of tokens for which we need to search for novel lexical
      entries.
    query_word_syntaxes: Possible syntactic categories for each of the query
      words. A dict mapping from token to set of CCG category instances.
    sentence: Token list sentence.
    ontology: Available logical ontology -- used to enumerate possible logical
      forms.
    model: Scene model which evaluates logical forms to answers.
    answer: Ground-truth answer to `sentence`.
  """

  # Target lexicon to be returned.
  lex = old_lex.clone()

  # TODO may overwrite
  for token in query_tokens:
    lex._entries[token] = []

  # TODO need to work on *product space* for multiple query words
  successes = defaultdict(list)
  for token in query_tokens:
    cand_syntaxes = query_token_syntaxes[token]
    for expr in ontology.iter_expressions(max_depth=6):
      for category in cand_syntaxes:
        if not is_compatible(category, expr):
          continue

        print(expr)
        lex._entries[token] = [Token(token, category, expr)]

        # Attempt a parse.
        results = chart.WeightedCCGChartParser(lex).parse(sentence)
        if results:
          # Parse succeeded -- check the candidate results.
          for result in results:
            # TODO skip re-checking parses with the same semantics
            semantics = result.label()[0].semantics()

            try:
              pred_answer = model.evaluate(semantics)
            except (TypeError, AttributeError):
              # Type inconsistency. TODO catch this in the iter_expression stage.
              continue
            except AssertionError:
              # Precondition of semantics failed to pass.
              continue

            if pred_answer == answer:
              # Parse succeeded with correct meaning. Add to the EC frontier.
              successes[token].append(Token(token, category, expr))

  for token in query_tokens:
    if not successes[token]:
      raise RuntimeError("Failed to derive any meanings for token %s." % token)
    lex._entries[token] = successes[token]

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
  parse_results = chart.WeightedCCGChartParser(lexicon).parse(sentence, True)

  valid_cands = [set() for _ in entry_idxs]
  for _, _, edge_cands in parse_results:
    for entry_idx, valid_cands_set in zip(entry_idxs, valid_cands):
      valid_cands_set.add(edge_cands[entry_idx])

  # Find valid interpretations across all uses of the word in the
  # sentence.
  valid_cands = list(reduce(lambda x, y: x & y, valid_cands))
  if not valid_cands:
    raise ValueError("no consistent interpretations of word found.")

  new_lex = lexicon.clone()
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

  lex = Lexicon.fromstring(r"""
  :- NN, DET, ADJ

  DET :: NN/NN
  ADJ :: NN/NN

  the => DET {\x.unique(x)}
  big => ADJ {\x.filter_size(x,big)}
  dog => NN {dog}""", include_semantics=True)
  print(augment_lexicon(lex, "the small dog".split(), "unique(filter_size(dog,small))"))
