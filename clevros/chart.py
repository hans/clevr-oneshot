import itertools

from nltk.ccg import chart as nchart
import numpy as np


printCCGDerivation = nchart.printCCGDerivation


class WeightedCCGChartParser(nchart.CCGChartParser):
  """
  CCG chart parser building off of the basic NLTK parser.

  Current extensions:

  1. Weighted inference (with weights on lexicon)
  2. Exhaustive search in cases where lexicon entries have ambiguous
  semantics. By default, NLTK ignores entries which have different
  semantics but share syntactic categories.
  """

  def __init__(self, lexicon, ruleset=None, *args, **kwargs):
    if ruleset is None:
      ruleset = nchart.DefaultRuleSet
    super().__init__(lexicon, ruleset, *args, **kwargs)

  def _parse_inner(self, chart):
    """
    Run a chart parse on a chart with the edges already filled in.
    """

    # Select a span for the new edges
    for span in range(2,chart.num_leaves()+1):
      for start in range(0,chart.num_leaves()-span+1):
        # Try all possible pairs of edges that could generate
        # an edge for that span
        for part in range(1,span):
          lstart = start
          mid = start + part
          rend = start + span

          for left in chart.select(span=(lstart,mid)):
            for right in chart.select(span=(mid,rend)):
              # Generate all possible combinations of the two edges
              for rule in self._rules:
                edges_added_by_rule = 0
                for newedge in rule.apply(chart,self._lexicon,left,right):
                  edges_added_by_rule += 1

    parses = chart.parses(self._lexicon.start())
    return parses

  def parse(self, tokens, return_aux=False):
    """
    Args:
      tokens: list of string tokens
      return_aux: return auxiliary information (`weights`, `valid_edges`)

    Returns:
      parses: list of CCG derivation results
      if return_aux, the list is actually a tuple with `parses` as its first
      element and the other following elements:
        weight: float parse weight
        edges: `tokens`-length list of the edge tokens used to generate this
          parse
    """
    tokens = list(tokens)
    lex = self._lexicon

    # Collect potential leaf edges for each index. May be multiple per
    # token.
    edge_cands = [[nchart.CCGLeafEdge(i, l_token, token) for l_token in lex.categories(token)]
            for i, token in enumerate(tokens)]

    # Run a parse for each of the product of possible leaf nodes,
    # and merge results.
    results = []
    used_edges = []
    for edge_sequence in itertools.product(*edge_cands):
      chart = nchart.CCGChart(list(tokens))
      for leaf_edge in edge_sequence:
        chart.insert(leaf_edge, ())

      partial_results = list(self._parse_inner(chart))
      results.extend(partial_results)

      if return_aux:
        # Track which edge values were used to generate these parses.
        used_edges.extend([edge_sequence] * len(partial_results))

    # Sort by weights derived from lexicon.
    def score_parse(parse):
      return sum(np.log(max(token.weight(), 1e-6)) for _, token in parse.pos())

    results = sorted(results, key=score_parse, reverse=True)
    if not return_aux:
      return results
    return [(parse, score_parse(parse), used_edges_i)
            for parse, used_edges_i in zip(results, used_edges)]

