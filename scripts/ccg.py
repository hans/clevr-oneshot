"""
CCG-style inference with fixed syntax and a fixed lexicon. (For now..)
"""

from nltk.ccg import chart, lexicon

question = "Are there any other things that are the same shape as the big metallic object?"
parse = "exist(same_shape(unique(filter_material(filter_size(scene, 'large'), 'metal'))))"

lex = lexicon.fromstring(r"""
    :- NN, INP, ADJ, DET, IN

    DET :: NN/NN
    ADJ :: NN/NN
    IN :: (NN\NN[comp])/NN

    same => ADJ {\x.same(x)}
    shape => NN {'shape'}
    as => IN {\x y.pair(x,y)}

    the => DET {\P.unique(P)}

    big => ADJ {\x.filter_size(x,'large')}
    metallic => ADJ {\x.filter_material(x,'metal')}

    object => NN {scene}""", include_semantics=True)

parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
results = list(parser.parse("the same shape as the big metallic object".split()))
for parse in results[:3]:
  chart.printCCGDerivation(parse)

