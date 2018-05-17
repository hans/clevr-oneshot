"""
Tools for updating and expanding lexicons, dealing with logical forms, etc.
"""

from collections import defaultdict, Counter
import copy
from functools import reduce
import itertools
import logging
import queue

from nltk.ccg import lexicon as ccg_lexicon
from nltk.ccg.api import PrimitiveCategory, FunctionalCategory, AbstractCCGCategory
from nltk.sem import logic as l
import numpy as np

from clevros import chart
from clevros.combinator import category_search_replace, \
    type_raised_category_search_replace
from clevros.clevr import scene_candidate_referents
from clevros.logic import get_arity


L = logging.getLogger(__name__)


class Lexicon(ccg_lexicon.CCGLexicon):

  def __init__(self, start, primitives, families, entries, ontology=None):
    """
    Create a new Lexicon.

    Args:
      start: Start symbol. All valid parses must have a root node of this
        category.
      primitives:
      families:
      entries: Lexical entries. Dict mapping from word strings to lists of
        `Token`s.
    """
    self._start = ccg_lexicon.PrimitiveCategory(start)
    self._primitives = primitives
    self._families = families
    self._entries = entries

    self.ontology = ontology

    self._derived_categories = {}
    self._derived_categories_by_base = defaultdict(set)
    self._derived_categories_by_source = {}

  @classmethod
  def fromstring(cls, lex_str, ontology=None, include_semantics=False):
    """
    Convert string representation into a lexicon for CCGs.
    """
    ccg_lexicon.CCGVar.reset_id()
    primitives = []
    families = {}
    entries = defaultdict(list)
    for line in lex_str.splitlines():
      # Strip comments and leading/trailing whitespace.
      line = ccg_lexicon.COMMENTS_RE.match(line).groups()[0].strip()
      if line == "":
        continue

      if line.startswith(':-'):
        # A line of primitive categories.
        # The first one is the target category
        # ie, :- S, N, NP, VP
        primitives = primitives + [prim.strip() for prim in line[2:].strip().split(',')]
      else:
        # Either a family definition, or a word definition
        (ident, sep, rhs) = ccg_lexicon.LEX_RE.match(line).groups()
        (catstr, semantics_str, weight) = ccg_lexicon.RHS_RE.match(rhs).groups()
        (cat, var) = ccg_lexicon.augParseCategory(catstr, primitives, families)

        if sep == '::':
          # Family definition
          # ie, Det :: NP/N
          families[ident] = (cat, var)
          # TODO weight?
        else:
          semantics = None
          if include_semantics is True:
            if semantics_str is None:
              raise AssertionError(line + " must contain semantics because include_semantics is set to True")
            else:
              semantics = l.Expression.fromstring(ccg_lexicon.SEMANTICS_RE.match(semantics_str).groups()[0])

          if weight is not None:
            weight = float(weight[1:-1])
          else:
            weight = 1.0

          # Word definition
          # ie, which => (N\N)/(S/NP)
          entries[ident].append(Token(ident, cat, semantics, weight))
    return cls(primitives[0], primitives, families, entries,
               ontology=ontology)

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

  def parse_category(self, cat_str):
    return ccg_lexicon.augParseCategory(cat_str, self._primitives, self._families)[0]

  @property
  def primitive_categories(self):
    return set([self.parse_category(prim) for prim in self._primitives])

  @property
  def observed_categories(self):
    """
    Find categories (both primitive and functional) attested in the lexicon.
    """
    return set([token.categ()
                for token_list in self._entries.values()
                for token in token_list])

  @property
  def category_semantic_arities(self):
    """
    Get the arities of semantic expressions associated with each observed
    syntactic category.
    """
    # If possible, lean on the type system to help determine expression arity.
    get_arity = (self.ontology and self.ontology.get_expr_arity) or get_arity

    entries_by_categ = {
      category: set(entry for entry in itertools.chain.from_iterable(self._entries.values())
                    if entry.categ() == category)
      for category in self.observed_categories
    }

    return {
      category: set(get_arity(entry.semantics()) for entry in entries)
      for category, entries in entries_by_categ.items()
    }

  def add_derived_category(self, involved_tokens, source_name=None):
    name = "D%i" % len(self._derived_categories)
    categ = DerivedCategory(name, involved_tokens[0].categ(),
                            source_name=source_name)
    self._primitives.append(categ)
    self._derived_categories[name] = (categ, set(involved_tokens))
    self._derived_categories_by_base[categ.base].add(categ)

    if source_name is not None:
      self._derived_categories_by_source[source_name] = categ

    return name

  def propagate_derived_category(self, name):
    categ, involved_entries = self._derived_categories[name]

    # Replace all lexical entries directly involved with the derived category.
    for entry_list in self._entries.values():
      for entry in entry_list:
        if entry in involved_entries:
          entry._categ = categ

    # Create duplicates of all entries with functional categories involving the
    # base of the derived category.
    #
    # For example, if we have an entry of syntactic category `S/N/PP` and we
    # have just created a derived category `D0` based on `N`, we need to make
    # sure there is now a corresponding candidate entry of type `S/D0/PP`.
    #
    # If we are using a functional category, also propagate the functional
    # category onto entries which have its partial yield. (We can
    # equivalently say that we are propagating onto instances using the
    # type-lowered version of this functional category.)
    #
    # For example, suppose the base category is `PP/NP` and there is an entry
    # with category `S/NP/PP`. The base category may participate in application
    # with the category `S/NP/PP` via composition, and so we would like to have
    # a type-lifted lexical entry involving the derived category which reflects
    # that fact.
    #
    # This is all a bit of a hack to make sure that we can tag certain lexical
    # entries as utilizing derived categories which are functional. There might
    # be other better longer-term implementations -- e.g. creating a custom
    # tagging mechanisms that accomplishes the same purpose.

    replacements = {}
    for word, entries in self._entries.items():
      new_entries = []

      for entry in entries:
        if not isinstance(entry.categ(), FunctionalCategory):
          # TODO will break with DerivedCategory cases
          continue

        try:
          categ_replacements = replacements[entry.categ()]
        except KeyError:
          if isinstance(categ.base, PrimitiveCategory):
            replacements[entry.categ()] = category_search_replace(
                entry.categ(), categ.base, categ)
          elif isinstance(categ.base, FunctionalCategory):
            replacements[entry.categ()] = type_raised_category_search_replace(
                entry.categ(), categ.base, categ)

          categ_replacements = replacements[entry.categ()]

        for replacement_category in categ_replacements:
          # We already know a replacement is necessary -- go ahead.
          new_entry = entry.clone()
          new_entry._categ = replacement_category
          new_entries.append(new_entry)

      self._entries[word] = entries + new_entries


  def lf_ngrams(self, order=1, condition_on_syntax=True, smooth=True):
    """
    Calculate n-gram statistics about the predicates present in the semantic
    forms in the lexicon.

    Args:
      order:
      condition_on_syntax: If `True`, returns a dict mapping each syntactic
        type to a different distribution over semantic predicates. If `False`,
        returns a single distribution.
      smooth: If `True`, add-1 smooth the returned distributions.
    """
    if order > 1:
      raise NotImplementedError()

    ret = defaultdict(Counter)
    for entry_list in self._entries.values():
      for entry in entry_list:
        for predicate in entry.semantics().predicates():
          key = entry.categ() if condition_on_syntax else None
          # TODO weight based on entry weight?
          ret[key][predicate.name] += 1

    if smooth:
      for key in ret:
        for predicate in ret[key]:
          ret[key][predicate] += 1
        ret[key][None] += 1

    # Normalize.
    ret_normalized = {}
    for categ in ret:
      Z = sum(ret[categ].values())
      ret_normalized[categ] = {word: count / Z
                               for word, count in ret[categ].items()}

    if not condition_on_syntax:
      return ret_normalized[None]
    return ret_normalized


class DerivedCategory(AbstractCCGCategory):

  def __init__(self, name, base, source_name=None):
    self.name = name
    self.base = base
    self.source_name = source_name
    self._comparison_key = (name, base)

  def is_primitive(self):
    return self.base.is_primitive()

  def is_function(self):
    return self.base.is_function()

  def is_var(self):
    return self.base.is_var()

  def categ(self):
    return self.base.categ()

  def substitute(self, subs):
    return self.base.substitute(subs)

  def can_unify(self, other):
    return self.base.can_unify(other)

  def arg(self):
    # exceptions in case is_primitive()
    return self.base.arg()

  def res(self):
    return self.base.res()

  def dir(self):
    return self.base.dir()

  def __str__(self):
    return "%s{%s}" % (self.name, self.base)

  def __repr__(self):
    return "%s{%s}{%s}" % (self.name, self.base, self.source_name)


class Token(ccg_lexicon.Token):

  def clone(self):
    return Token(self._token, self._categ, self._semantics, self._weight)

  def __str__(self):
    return "Token(%s => %s%s)" % (self._token, self._categ,
                                  " {%s}" % self._semantics if self._semantics else "")

  __repr__ = __str__


def get_semantic_arity(category, arity_overrides=None):
  """
  Get the expected arity of a semantic form corresponding to some syntactic
  category.
  """
  arity_overrides = arity_overrides or {}
  if category in arity_overrides:
    return arity_overrides[category]

  if isinstance(category, DerivedCategory):
    return get_semantic_arity(category.base, arity_overrides)
  elif isinstance(category, PrimitiveCategory):
    return 0
  elif isinstance(category, FunctionalCategory):
    return 1 + get_semantic_arity(category.arg(), arity_overrides) \
      + get_semantic_arity(category.res(), arity_overrides)
  else:
    raise ValueError("unknown category type %r" % category)


def get_yield(category):
  """
  Get the root node of a syntactic category.
  """
  if isinstance(category, DerivedCategory):
    if isinstance(category.base, PrimitiveCategory):
      return category.categ()
    else:
      return get_yield(category.base)
  elif isinstance(category, PrimitiveCategory):
    return category.categ()
  elif isinstance(category, FunctionalCategory):
    if category.dir().is_forward():
      return get_yield(category.res())
    else:
      return get_yield(category.arg())
  else:
    raise ValueError("unknown category type with instance %r" % category)


def get_candidate_categories(lex, tokens, sentence):
  """
  Find candidate categories for the given tokens which appear in `sentence` such
  that `sentence` yields a parse.

  Args:
    lex:
    tokens:
    sentence

  Returns:
    sorted_token_cat_weights: Dictionary mapping each token to a ranking over
      candidate categories. The dictionary values are a weighted list with
      elements of form `(category, weight)`, sorted by descending weight.
  """
  assert set(tokens).issubset(set(sentence))

  # Make a minimal copy of `lex` which does not track semantics.
  lex = lex.clone(retain_semantics=False)

  # Remove entries for the queried tokens.
  for token in tokens:
    lex._entries[token] = []

  candidate_categories = lex.observed_categories

  def evaluate_cat_assignment(cat_assignment):
    for token, category in zip(tokens, cat_assignment):
      lex._entries[token] = [Token(token, category)]

    # Attempt a parse.
    results = chart.WeightedCCGChartParser(lex, chart.DefaultRuleSet) \
        .parse(sentence)
    if results:
      # Prior weight for category comes from lexicon.
      #
      # Might also downweight categories which require type-lifting parses by
      # default?
      score = 0.0
      for token, category in zip(tokens, cat_assignment):
        category_score = sum(entry.weight() for entries in lex._entries.values()
                             for entry in entries if entry.categ() == category)
        score += category_score

      # Likelihood weight comes from parse score?
      # return sum(weight for _, weight, _ in results)

      return score

    return 0.0

  # NB does not cover the case where a single token needs multiple syntactic
  # interpretations for the sentence to parse
  cat_assignment_weights = {
    cat_assignment: evaluate_cat_assignment(cat_assignment)
    for cat_assignment in itertools.product(candidate_categories, repeat=len(tokens))
  }

  token_cat_weights = defaultdict(Counter)
  for cat_assignment, weight in cat_assignment_weights.items():
    for token, token_cat_assignment in zip(tokens, cat_assignment):
      token_cat_weights[token][token_cat_assignment] += weight

  # Normalize.
  sorted_token_cat_weights = {}
  for token, weighted_list in token_cat_weights.items():
    Z = sum(weighted_list.values())
    weighted_list = [(cat, weight / Z) for cat, weight in weighted_list.items()]
    weighted_list = sorted(weighted_list, reverse=True, key=lambda x: x[1])
    sorted_token_cat_weights[token] = weighted_list

  return sorted_token_cat_weights


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

  # Prepare for syntactic bootstrap: pre-calculate distributions over semantic
  # form elements conditioned on syntactic category.
  lf_ngrams = lex.lf_ngrams(order=1, condition_on_syntax=True, smooth=True)

  # We will restrict semantic arities based on the observed arities available
  # for each category. Pre-calculate the necessary associations.
  category_sem_arities = lex.category_semantic_arities

  # Enumerate expressions just once! We'll bias the search over the enumerated
  # forms later.
  candidate_exprs = set(ontology.iter_expressions(max_depth=3))

  # TODO need to work on *product space* for multiple query words
  successes = defaultdict(set)
  semantics_results = {}
  for token in query_tokens:
    candidate_queue = queue.PriorityQueue(maxsize=100)
    category_parse_results = {}

    # Prepare dummy variable which will be inserted into parse checks.
    sub_target = l.Variable("F0000")
    sub_expr = l.FunctionVariableExpression(sub_target)

    cand_syntaxes = query_token_syntaxes[token]
    L.info("Candidate syntaxes for %s: %r", token, cand_syntaxes)
    for category, category_weight in cand_syntaxes:
      # Prepare to BOOTSTRAP: Bias expression iteration based on the syntactic
      # category.
      cat_lf_ngrams = lf_ngrams[category]
      # Redistribute UNK probability uniformly across predicates not observed
      # for this category.
      unk_lf_prob = cat_lf_ngrams.pop(None)
      unobserved_preds = set(f.name for f in ontology.functions) - set(cat_lf_ngrams.keys())
      cat_lf_ngrams.update({pred: unk_lf_prob / len(unobserved_preds)
                           for pred in unobserved_preds})

      print(category, ", ".join("%.03f %s" % (prob, pred) for pred, prob
                                in sorted(cat_lf_ngrams.items(), key=lambda x: x[1], reverse=True)))

      # Parse just once with a dummy variable in the place of the candidate
      # semantics.
      #
      # We'll next enumerate all possible expressions, substitute in for this
      # dummy variable, and then attempt to evaluate.
      lex._entries[token] = [Token(token, category, sub_expr)]
      # TODO this only works for basic application right now -- we can't yet
      # support composition with the dummy variable setup. Need to extract some
      # lambdas from the dummy variable s.t. the parser can work its
      # composition magic.
      results = chart.WeightedCCGChartParser(lex, ruleset=chart.ApplicationRuleSet) \
          .parse(sentence)
      category_parse_results[category] = results

      for expr in candidate_exprs:
        if get_arity(expr) not in category_sem_arities[category]:
          # TODO rather than arity-checking post-hoc, form a type request
          continue

        likelihood = 0.0
        for predicate in expr.predicates():
          if predicate.name in cat_lf_ngrams:
            likelihood += np.log(cat_lf_ngrams[predicate.name])

        joint_score = np.log(category_weight) + likelihood
        new_item = (joint_score, (category, expr))
        try:
          candidate_queue.put_nowait(new_item)
        except queue.Full:
          # See if this candidate is better than the worst item.
          worst = candidate_queue.get()
          if worst[0] < joint_score:
            replacement = new_item
          else:
            replacement = worst

          candidate_queue.put_nowait(replacement)

    # NB not parallelizing anything below
    candidates = sorted(candidate_queue.queue,
                        key=lambda item: -item[0])
    for joint_score, (category, expr) in candidates:
      parse_results = category_parse_results[category]

      # Parse succeeded -- check the candidate results.
      for result in parse_results:
        semantics = result.label()[0].semantics()
        semantics = semantics.replace(sub_target, expr)

        # Check cached result first.
        success = semantics_results.get(semantics, None)
        if success is None:
          # Evaluate the expression and cache result.
          try:
            pred_answer = model.evaluate(semantics)
            success = pred_answer == answer
          except (TypeError, AttributeError) as e:
            # Type inconsistency. TODO catch this in the iter_expression
            # stage, or typecheck before evaluating.
            success = False
          except AssertionError as e:
            # Precondition of semantics failed to pass.
            success = False

          # Cache evaluation result.
          semantics_results[semantics] = success

        if success:
          # Parse succeeded with correct meaning. Add candidate lexical entry.
          successes[token].add((joint_score, (category, expr)))

  for token in query_tokens:
    try:
      successes_t = list(successes[token])
    except KeyError:
      raise RuntimeError("Failed to derive any meanings for token %s." % token)
    if not successes_t:
      raise RuntimeError("Failed to derive any meanings for token %s." % token)

    # Compute weights for competing entries by a stable softmax.
    weights_t = np.array([weight for weight, _ in successes_t])
    weights_t -= weights_t.max()
    weights_t = np.exp(weights_t)
    weights_t /= weights_t.sum()

    lex._entries[token] = [Token(token, category, expr, weight=softmax_weight)
                           for (_, (category, expr)), softmax_weight
                           in zip(successes_t, weights_t)]

    # DEBUG
    L.debug("Inferred %i novel entries for token %s:", len(successes_t), token)
    for (_, entry_info), weight in zip(successes_t, weights_t):
      L.debug("%.4f %s", weight, entry_info)

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
  expr = l.Expression.fromstring(lf_str)
  assert isinstance(expr, l.ApplicationExpression)

  # First candidates: all available constants
  candidates = set([l.ConstantExpression(const)
                    for const in expr.constants()])

  # All level-1 abstractions of the LF
  queue = [expr]
  while queue:
    node = queue.pop()

    n_constants = 0
    for arg in node.args:
      if isinstance(arg, l.ConstantExpression):
        n_constants += 1
      elif isinstance(arg, l.ApplicationExpression):
        queue.append(arg)
      else:
        assert False, "Unexpected type " + str(arg)

    # Hard constraint for now: all but one variable should be a
    # constant expression.
    if n_constants < len(node.args) - 1:
      continue

    # Create the candidate node(s).
    variable = l.Variable("x")
    for i, arg in enumerate(node.args):
      if isinstance(arg, l.ApplicationExpression):
        new_arg_cands = [l.VariableExpression(variable)]
      else:
        new_arg_cands = [arg]
        if n_constants == len(node.args):
          # All args are constant, so turning each constant into
          # a variable is also legal. Do that.
          new_arg_cands.append(l.VariableExpression(variable))

      # Enumerate candidate new arguments and yield candidate new exprs.
      for new_arg_cand in new_arg_cands:
        new_args = node.args[:i] + [new_arg_cand] + node.args[i + 1:]
        app_expr = l.ApplicationExpression(node.pred, new_args[0])
        app_expr = reduce(lambda x, y: l.ApplicationExpression(x, y), new_args[1:], app_expr)
        candidates.add(l.LambdaExpression(variable, app_expr))

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
