import logging

from clevros.compression import Compressor
from clevros.lexicon import augment_lexicon_distant, predict_zero_shot, \
    get_candidate_categories, get_semantic_arity
from clevros.perceptron import update_perceptron_distant


L = logging.getLogger(__name__)


class WordLearner(object):

  def __init__(self, lexicon, compressor, bootstrap=True,
               learning_rate=10.0, beta=3.0, negative_samples=5,
               total_negative_mass=0.1, syntax_prior_smooth=1e-3,
               meaning_prior_smooth=1e-3, bootstrap_alpha=0.25):

    """
    Args:
      lexicon:
      compressor:
      bootstrap: If `True`, enable syntactic bootstrapping.
    """
    self.lexicon = lexicon
    self.compressor = compressor

    self.bootstrap = bootstrap

    # Learning hyperparameters
    self.learning_rate = learning_rate
    self.beta = beta
    self.negative_samples = negative_samples
    self.total_negative_mass = total_negative_mass
    self.syntax_prior_smooth = syntax_prior_smooth
    self.meaning_prior_smooth = meaning_prior_smooth
    self.bootstrap_alpha = bootstrap_alpha

  @property
  def ontology(self):
    return self.lexicon.ontology

  def compress_lexicon(self):
    if self.compressor is None:
      return

    # Run EC compression on the entries of the induced lexicon. This may create
    # new inventions, updating both the `ontology` and the provided `lex`.
    new_lex, affected_entries = self.compressor.make_inventions(self.lexicon)

    # Create derived categories following new inventions.
    to_propagate = []
    for invention_name, tokens in affected_entries.items():
      if invention_name in new_lex._derived_categories_by_source:
        # TODO merge possibly new tokens with existing invention token groups
        continue

      affected_syntaxes = set(t.categ() for t in tokens)
      if len(affected_syntaxes) == 1:
        # Just one syntax is involved. Create a new derived category.
        L.debug("Creating new derived category for tokens %r", tokens)

        derived_name = new_lex.add_derived_category(tokens, source_name=invention_name)
        to_propagate.append((derived_name, next(iter(affected_syntaxes))))


    # Propagate derived categories, beginning with the largest functional
    # categories. The ordering allows us to support hard-propagating both e.g.
    # a new root category and a new argument category at the same time.
    #
    # (We may have derived new categories for entries with types `PP/NP` and # `S/NP/PP` -- in this case, we want to first make available a new category
    # `D0{S}/NP/PP` such that we can propagate the derived `D1{PP}` onto it,
    # yielding `D0{S}/NP/D1{PP}`.)
    to_propagate = sorted(
        to_propagate, key=lambda proposal: get_semantic_arity(proposal[1]),
        reverse=True)
    for derived_cat, base in to_propagate:
      new_lex.propagate_derived_category(derived_cat)
      cat_obj, _ = new_lex._derived_categories[derived_cat]
      L.info("Propagated derived category %s (source %s)", cat_obj, cat_obj.source_name)

    self.lexicon = new_lex

  def prepare_lexical_induction(self, sentence):
    """
    Find the tokens in a sentence which need to be updated such that the
    sentence will parse.

    Args:
      sentence: Sequence of tokens

    Returns:
      query_tokens: List of tokens which need to be updated
      query_token_syntaxes: Dict mapping tokens to weighted list of candidate
        syntaxes (as returned by `get_candidate_categoies`)
    """
    query_tokens = [word for word in sentence
                    if not self.lexicon._entries.get(word, [])]
    if len(query_tokens) > 0:
      # Missing lexical entries -- induce entries for words which clearly
      # require an entry inserted
      L.info("Novel words: %s", " ".join(query_tokens))
      query_token_syntaxes = get_candidate_categories(
          self.lexicon, query_tokens, sentence,
          smooth=self.syntax_prior_smooth)

      return query_tokens, query_token_syntaxes

    # Lexical entries are present for all words, but parse still failed.
    # That means we are missing entries for one or more wordforms.
    # For now: blindly try updating each word's entries.
    #
    # TODO: Does not handle case where multiple words need an update.
    query_tokens, query_token_syntaxes = [], []
    for token in sentence:
      query_tokens = [token]
      query_token_syntaxes = get_candidate_categories(
          self.lexicon, query_tokens, sentence,
          smooth=self.syntax_prior_smooth)

      if query_token_syntaxes:
        # Found candidate parses! Let's try adding entries for this token,
        # then.
        return query_tokens, query_token_syntaxes

    raise ValueError(
        "unable to find new entries which will make the sentence parse: %s" % sentence)

  def predict_zero_shot(self, sentence):
    """
    Yield zero-shot predictions on the syntax and meaning of words in the
    sentence requiring novel lexical entries.

    Args:
      sentence: List of token strings

    Returns:
      syntaxes: Dict mapping tokens to posterior distributions over syntactic
        categories
      joint_candidates: Dict mapping tokens to posterior distributions over
        tuples `(syntax, lf)`
    """
    # Find tokens for which we need to insert lexical entries.
    query_tokens, query_token_syntaxes = self.prepare_lexical_induction(sentence)
    candidates, _, _ = predict_zero_shot(
        self.lexicon, query_tokens, query_token_syntaxes,
        sentence, self.ontology,
        bootstrap=self.bootstrap,
        meaning_prior_smooth=self.meaning_prior_smooth,
        alpha=self.bootstrap_alpha)
    return query_token_syntaxes, candidates

  def update_with_example(self, sentence, model, answer):
    """
    Observe a new `sentence -> answer` pair in the context of some `model` and
    update learner weights.

    Args:
      sentence: List of token strings
      model: `Model` instance
      answer: Desired result from `model.evaluate(lf_result(sentence))`

    Returns:
      weighted_results: List of weighted parse results for the example.
    """

    try:
      weighted_results, _ = update_perceptron_distant(
          self.lexicon, sentence, model, answer,
          learning_rate=self.learning_rate)
    except ValueError as e:
      # No parse succeeded -- attempt lexical induction.
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))
      L.warning(e)

      # Find tokens for which we need to insert lexical entries.
      query_tokens, query_token_syntaxes = \
          self.prepare_lexical_induction(sentence)
      L.info("Inducing new lexical entries for words: %s", ", ".join(query_tokens))

      # Augment the lexicon with all entries for novel words which yield the
      # correct answer to the sentence under some parse. Restrict the search by
      # the supported syntaxes for the novel words (`query_token_syntaxes`).
      self.lexicon = augment_lexicon_distant(
          self.lexicon, query_tokens, query_token_syntaxes, sentence,
          self.ontology, model, answer,
          bootstrap=self.bootstrap,
          meaning_prior_smooth=self.meaning_prior_smooth,
          alpha=self.bootstrap_alpha, beta=self.beta,
          negative_samples=self.negative_samples,
          total_negative_mass=self.total_negative_mass)

      self.lexicon.debug_print()

      # Compress the resulting lexicon.
      self.compress_lexicon()

      # Attempt a new parameter update.
      weighted_results, _ = update_perceptron_distant(
          self.lexicon, sentence, model, answer,
          learning_rate=self.learning_rate)

    prune_count = self.lexicon.prune()
    L.info("Pruned %i entries from lexicon.", prune_count)

    return weighted_results
