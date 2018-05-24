import logging


L = logging.getLogger(__name__)


class WordLearner(object):

  def __init__(self, lexicon, compressor, bootstrap=True):
    self.lexicon = lexicon
    self.ontology = self.lexicon.ontology
    self.compressor = compressor

    self.bootstrap = bootstrap

  def compress_lexicon(self):
    if self.compressor is None:
      return

    # Run EC compression on the entries of the induced lexicon. This may create
    # new inventions, updating both the `ontology` and the provided `lex`.
    new_lex, affected_entries = self.compressor.make_inventions(self.lexicon)

    for invention_name, tokens in affected_entries.items():
      if invention_name in new_lex._derived_categories_by_source:
        # TODO merge possibly new tokens with existing invention token groups
        continue

      affected_syntaxes = set(t.categ() for t in tokens)
      if len(affected_syntaxes) == 1:
        # Just one syntax is involved. Create a new derived category.
        L.debug("Creating new derived category for tokens %r", tokens)

        derived_name = new_lex.add_derived_category(tokens, source_name=invention_name)
        new_lex.propagate_derived_category(derived_name)

        L.info("Created and propagated derived category %s == %s -- %r",
              derived_name, new_lex._derived_categories[derived_name][0].base, tokens)

    self.lexicon = new_lex

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
      weighted_results, _ = update_perceptron_distant(self.lexicon, sentence,
                                                      model, answer)
    except ValueError:
      # No parse succeeded -- attempt lexical induction.
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))

      query_tokens = [word for word in sentence
                      if not self.lexicon._entries.get(word, [])]
      L.info("Novel words: %s", " ".join(query_tokens))
      query_token_syntaxes = get_candidate_categories(
          self.lexicon, query_tokens, sentence)

      # Augment the lexicon with all entries for novel words which yield the
      # correct answer to the sentence under some parse. Restrict the search by
      # the supported syntaxes for the novel words (`query_token_syntaxes`).
      self.lexicon = augment_lexicon_distant(
          self.lexicon, query_tokens, query_token_syntaxes, sentence,
          self.ontology, model, answer, bootstrap=self.bootstrap)

      # Compress the resulting lexicon.
      self.compress_lexicon()

      # Attempt a new parameter update.
      weighted_results, _ = update_perceptron_distant(
          self.lexicon, sentence, model, answer)

      return weighted_results
