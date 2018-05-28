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

  def total_category_masses(self, exclude_tokens=frozenset(),
                            soft_propagate_roots=False):
    """
    Return the total weight mass assigned to each syntactic category.

    Args:
      exclude_tokens: Exclude entries with this token from the count.
      soft_propagate_roots: Soft-propagate derived root categories. If there is
        a derived root category `D0{S}` and some lexical entry `S/N`, even if
        no entry has the category `D0{S}`, we will add a key to the returned
        counter with category `D0{S}/N` (and zero weight).
    """
    ret = Counter()
    # Track categories with root yield.
    rooted_cats = set()

    for token, entries in self._entries.items():
      if token in exclude_tokens:
        continue
      for entry in entries:
        if get_yield(entry.categ()) == self._start:
          rooted_cats.add(entry.categ())

        ret[entry.categ()] += entry.weight()

    if soft_propagate_roots:
      derived_root_cats = self._derived_categories_by_base[self._start]
      for rooted_cat in rooted_cats:
        for derived_root_cat in derived_root_cats:
          soft_prop_cat = set_yield(rooted_cat, derived_root_cat)
          ret.setdefault(soft_prop_cat, 0.0)

    return ret

  def observed_category_distribution(self, exclude_tokens=frozenset(),
                                     soft_propagate_rootes=False):
    """
    Return a distribution over categories calculated using the lexicon weights.
    """
    ret = self.total_category_masses(exclude_tokens=exclude_tokens)
    Z = sum(ret.values())
    if Z > 0:
      ret = {category: weight / Z for category, weight in ret.items()}

    return ret

  @property
  def start_categories(self):
    """
    Return primitive categories which are valid root nodes.
    """
    return [self._start] + list(self._derived_categories_by_base[self._start])

  def start(self):
    raise NotImplementedError("use #start_categories instead.")

  @property
  def category_semantic_arities(self):
    """
    Get the arities of semantic expressions associated with each observed
    syntactic category.
    """
    # If possible, lean on the type system to help determine expression arity.
    get_arity = (self.ontology and self.ontology.get_expr_arity) \
        or get_semantic_arity

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

    # The derived category will have as its base the yield of the source
    # category. For example, tokens of type `PP/NP` will lead to a derived
    # category of type `PP`.
    categ = DerivedCategory(name, get_yield(involved_tokens[0].categ()),
                            source_name=source_name)
    self._primitives.append(categ)
    self._derived_categories[name] = (categ, set(involved_tokens))
    self._derived_categories_by_base[categ.base].add(categ)

    if source_name is not None:
      self._derived_categories_by_source[source_name] = categ

    return name

  def propagate_derived_category(self, name):
    categ, involved_entries = self._derived_categories[name]
    originating_category = next(iter(involved_entries)).categ()
    new_entries = defaultdict(list)

    # Replace all lexical entries directly involved with the derived category.
    for word, entry_list in self._entries.items():
      for entry in entry_list:
        if entry in involved_entries:
          # Replace the yield of the syntactic category with our new derived
          # category. (For primitive categories, setting the yield is
          # equivalent to just changing the category.)
          new_entry = entry.clone()
          new_entry._categ = set_yield(entry.categ(), categ)
          new_entries[word].append(new_entry)

    # Create duplicates of all entries with functional categories involving the
    # base of the derived category.
    #
    # For example, if we have an entry of syntactic category `S/N/PP` and we
    # have just created a derived category `D0` based on `N`, we need to make
    # sure there is now a corresponding candidate entry of type `S/D0/PP`.

    replacements = {}
    # HACK: don't propagate S
    if categ.base.categ() != "S":
      for word, entries in self._entries.items():
        for entry in entries:
          if not isinstance(entry.categ(), FunctionalCategory):
            # This step only applies to functional categories.
            continue
          elif entry.categ() == originating_category:
            # We've found an entry which has a category with the same category
            # as that of the tokens involved in this derived category's
            # creation. Don't propagate -- this is exactly what allows us to
            # separate the involved tokens from other members of the same
            # category.
            #
            # e.g. if we've just derived a category from some PP/NP entries,
            # don't propagate the PP yield onto other PP/NP entries which were
            # not involved in the derived category creation.
            continue

          try:
            categ_replacements = replacements[entry.categ()]
          except KeyError:
            replacements[entry.categ()] = category_search_replace(
                entry.categ(), categ.base, categ)

            categ_replacements = replacements[entry.categ()]

          for replacement_category in categ_replacements:
            # We already know a replacement is necessary -- go ahead.
            new_entry = entry.clone()
            new_entry._categ = replacement_category
            new_entries[word].append(new_entry)

    for word, w_entries in new_entries.items():
      self._entries[word].extend(w_entries)


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

    ret = {}
    for entry_list in self._entries.values():
      for entry in entry_list:
        key = entry.categ() if condition_on_syntax else None
        # Initialize the distribution, whether or not we will find any
        # predicates to count.
        if key not in ret:
          ret[key] = Counter()

        for predicate in entry.semantics().predicates():
          # TODO weight based on entry weight?
          ret[key][predicate.name] += 1

    # TODO this isn't add-1 smoothing -- should collect the predicate support
    # *across* distributions and smooth using that.
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


class DerivedCategory(PrimitiveCategory):

  def __init__(self, name, base, source_name=None):
    self.name = name
    self.base = base
    self.source_name = source_name
    self._comparison_key = (name, base)

  def is_primitive(self):
    return True

  def is_function(self):
    return False

  def is_var(self):
    return False

  def categ(self):
    return self.name

  def substitute(self, subs):
    return self

  def can_unify(self, other):
    # The unification logic is critical here -- this determines how derived
    # categories are treated relative to their base categories.
    if other == self:
      return []

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
  Get the primitive yield node of a syntactic category.
  """
  if isinstance(category, DerivedCategory):
    if isinstance(category.base, PrimitiveCategory):
      return category.base
    else:
      return get_yield(category.base)
  elif isinstance(category, PrimitiveCategory):
    return category
  elif isinstance(category, FunctionalCategory):
    if category.dir().is_forward():
      return get_yield(category.res())
    else:
      return get_yield(category.arg())
  else:
    raise ValueError("unknown category type with instance %r" % category)


def set_yield(category, new_yield):
  if isinstance(category, PrimitiveCategory):
    return new_yield
  elif isinstance(category, FunctionalCategory):
    if category.dir().is_forward():
      res = set_yield(category.res(), new_yield)
      arg = category.arg()
    else:
      res = category.res()
      arg = set_yield(category.arg(), new_yield)

    return FunctionalCategory(res, arg, category.dir())
  else:
    raise ValueError("unknown category type of instance %r" % category)


def get_candidate_categories(lex, tokens, sentence, smooth=True):
  """
  Find candidate categories for the given tokens which appear in `sentence` such
  that `sentence` yields a parse.

  Args:
    lex:
    tokens:
    sentence:
    smooth: If `True`, add-1 smooth the returned distributions.

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

  category_prior = lex.observed_category_distribution(exclude_tokens=set(tokens))
  L.debug(category_prior)

  def score_cat_assignment(cat_assignment):
    """
    Assign a log-probability to a joint assignment of categories to tokens.
    """

    for token, category in zip(tokens, cat_assignment):
      lex._entries[token] = [Token(token, category)]

    # Attempt a parse.
    results = chart.WeightedCCGChartParser(lex, chart.DefaultRuleSet) \
        .parse(sentence, return_aux=True)
    if results:
      # Prior weight for category comes from lexicon.
      #
      # Might also downweight categories which require type-lifting parses by
      # default?
      score = 0
      for token, category in zip(tokens, cat_assignment):
        score += np.log(category_prior[category])
        pass

      # Likelihood weight comes from parse score
      score += np.log(sum(np.exp(weight)
                          for _, weight, _ in results))

      return score

    return -np.inf

  # NB does not cover the case where a single token needs multiple syntactic
  # interpretations for the sentence to parse
  cat_assignment_weights = {
    cat_assignment: score_cat_assignment(cat_assignment)
    for cat_assignment in itertools.product(category_prior.keys(), repeat=len(tokens))
  }

  token_cat_weights = defaultdict(Counter)
  if smooth:
    for token in tokens:
      for cat in category_prior.keys():
        token_cat_weights[token][cat] += 1e-4 # TODO do this in a principled way

  for cat_assignment, logp in cat_assignment_weights.items():
    for token, token_cat_assignment in zip(tokens, cat_assignment):
      token_cat_weights[token][token_cat_assignment] += np.exp(logp)

  # Normalize.
  sorted_token_cat_weights = {}
  for token, weighted_list in token_cat_weights.items():
    if len(weighted_list) == 0:
      raise RuntimeError("No valid syntactic categories available for token %s" % token)
    Z = max(sum(weighted_list.values()), 1e-5)

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


def attempt_candidate_parse(lexicon, token, candidate_category,
                            candidate_expressions, sentence, dummy_var=None):
  """
  Attempt to parse a sentence, mapping `token` to a new candidate lexical
  entry.

  Arguments:
    lexicon: Current lexicon. Will modify in place -- send a copy.
    token: String token to be attempted.
    candidate_category:
    candidate_expressions: Collection of candidate semantic forms.
    sentence: Sentence which we are attempting to parse.
  """

  # Prepare dummy variable which will be inserted into parse checks.
  sub_target = dummy_var or l.Variable("F000")
  sub_expr = l.FunctionVariableExpression(sub_target)

  lexicon._entries[token] = [Token(token, candidate_category, sub_expr)]

  parse_results = []

  # First attempt a parse with only function application rules.
  results = chart.WeightedCCGChartParser(lexicon, ruleset=chart.ApplicationRuleSet) \
      .parse(sentence)
  if results:
    # TODO: also attempt parses with composition?
    return results, sub_target

  # Attempt to parse, allowing for function composition. In order to support
  # this we need to pass a dummy expression which is a lambda.
  arities = {expr: get_arity(expr) for expr in candidate_expressions}
  max_arity = max(arities.values())

  results, sub_expr_original = [], sub_expr
  for arity in range(1, max(arities.values()) + 1):
    sub_expr = sub_expr_original

    variables = [l.Variable("z%i" % (i + 1)) for i in range(arity)]
    # Build curried application expression.
    term = sub_expr
    for variable in variables:
      term = l.ApplicationExpression(term, l.IndividualVariableExpression(variable))

    # Build surrounding lambda expression.
    sub_expr = term
    for variable in variables[::-1]:
      sub_expr = l.LambdaExpression(variable, sub_expr)

    lexicon._entries[token] = [Token(token, candidate_category, sub_expr)]
    results.extend(
        chart.WeightedCCGChartParser(lexicon, ruleset=chart.DefaultRuleSet).parse(sentence))

  # TODO need to return more information if we've lifted the sub_target
  return results, sub_target


def augment_lexicon_distant(old_lex, query_tokens, query_token_syntaxes,
                            sentence, ontology, model, answer,
                            bootstrap=True):
  """
  Augment a lexicon with candidate meanings for a given word using distant
  supervision. (The induced meanings for the queried words must yield parses
  that lead to `answer` under the `model`.)

  Candidate entries will be assigned relative weights according to a posterior
  distribution $P(word -> syntax, meaning | sentence, answer, lexicon)$. This
  distribution incorporates multiple prior and likelihood terms:

  1. A prior over syntactic categories (derived by inspection of the current
     lexicon)
  2. A likelihood over syntactic categories (derived by ranking candidate
     parses conditioning on each syntactic category)
  3. A prior over the predicates present in meaning representations,
     conditioned on syntactic category choice ("syntactic bootstrapping")

  The first two components are accomplished by `get_candidate_categories`,
  while the third is calculated within this function. (The third can be
  parametrically disabled using the `bootstrap` argument.)

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
    bootstrap: If `True`, incorporate priors over LF predicates conditioned on
      syntactic category when scoring candidate lexical entries.
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

  # Shared dummy variable which is included in candidate semantic forms, to be
  # replaced by all candidate lexical expressions and evaluated.
  dummy_var = None

  # TODO need to work on *product space* for multiple query words
  successes = defaultdict(set)
  semantics_results = {}
  for token in query_tokens:
    candidate_queue = queue.PriorityQueue(maxsize=500)
    category_parse_results = {}

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

      L.debug("% 20s %s", category,
              ", ".join("%.03f %s" % (prob, pred) for pred, prob
                        in sorted(cat_lf_ngrams.items(), key=lambda x: x[1], reverse=True)))

      # Attempt to parse with this parse category, and return the resulting
      # syntactic parses + sentence-level semantic forms, with a dummy variable
      # in place of where the candidate expressions will go.
      results, dummy_var = attempt_candidate_parse(lex, token, category, candidate_exprs,
                                                   sentence, dummy_var=dummy_var)
      category_parse_results[category] = results

      for expr in candidate_exprs:
        if get_arity(expr) not in category_sem_arities[category]:
          # TODO rather than arity-checking post-hoc, form a type request
          continue

        likelihood = 0.0
        if bootstrap:
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
        # TODO can we pre-compute the simplification?
        semantics = semantics.replace(dummy_var, expr).simplify()

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
