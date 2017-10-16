"""
CCG-style inference with fixed syntax and a fixed lexicon. (For now..)
"""

from nltk.ccg import chart, lexicon

question = "Are there any other things that are the same shape as the big metallic object?"
parse = "exist(same_shape(unique(filter_material(filter_size(scene, 'large'), 'metal'))))"

semantics = True

lex2 = lexicon.fromstring(r"""
    :- NN, INP, ADJ, DET, IN

    DET :: NN/NN
    ADJ :: NN/NN
    IN :: (NN\NN)/NN
    ADV :: NN/ADJ
    QS :: NN/NN

    same => ADJ {\x.same_(x)}

    same => IN {\x y.same_(x,y)}


    material => NN {'material'}
    color => NN {'color'}
    shape => NN {'shape'}
    size => NN {'size'}

    as => IN {\x y.as(x,y)}
    of => IN {\x y.of(x,y)}

    to => DET {\x.relate(x)}

    the => DET {\P.(P)}
    a => DET {\P.(P)}
    an => DET {\P.(P)}
    any => DET {\P.(P)}


    metallic => ADJ {\x.filter_material(x,'metal')}
    metal => ADJ {\x.filter_material(x,'metal')}
    shiny => ADJ {\x.filter_material(x,'metal')}
    rubber => ADJ {\x.filter_material(x,'rubber')}
    matte => ADJ {\x.filter_material(x,'rubber')}

    gray => ADJ {\x.filter_color(x,'gray')}
    red => ADJ {\x.filter_color(x,'red')}
    blue => ADJ {\x.filter_color(x,'blue')}
    green => ADJ {\x.filter_color(x,'green')}
    brown => ADJ {\x.filter_color(x,'brown')}
    purple => ADJ {\x.filter_color(x,'purple')}
    cyan => ADJ {\x.filter_color(x,'cyan')}
    yellow => ADJ {\x.filter_color(x,'yellow')}

    big => ADJ {\x.filter_size(x,'large')}
    large => ADJ {\x.filter_size(x,'large')}
    small => ADJ {\x.filter_size(x,'small')}
    tiny => ADJ {\x.filter_size(x,'small')}

    left => ADJ {\x.left(x,'prueba')}
    left => NN {'left'}

    cube => NN {'cube'}
    block => NN {'cube'}
    sphere => ADJ {\x.filter_shape(x,'sphere')}
    spheres => NN {'sphere'}
    ball => NN {'sphere'}
    cylinder => NN {'cylinder'}

    what_is => QS {\x.query_(x)}
    
    are_there => QS {\x.query_(x)}
    is_there => QS {\x.query_(x)}


    
    object => NN {scene}
    thing => NN {scene}
    it => NN {scene}

    """, include_semantics=semantics)

#TODO: 
#Left, right, etc
#that
#Hard one:
    #Is the purple thing the same shape as the large gray rubber thing?
    #equal_shape(query_shape(unique(filter_color(scene,u'purple'))),query_shape(unique(filter_material(filter_color(filter_size(scene,u'large'),u'gray'),u'rubber'))))
parser = chart.CCGChartParser(lex2, chart.DefaultRuleSet)

#results = list(parser.parse("the same shape as the big metallic object".split()))
#results = list(parser.parse("a big brown object of the same shape as the green thing".split()))
results = list(parser.parse("the material of the big purple object".split()))
#results = list(parser.parse("any sphere to the left of it".split()))
#results = list(parser.parse("the purple thing the same shape as the large gray rubber thing".split()))

chart.printCCGDerivation(results[0])


#are there any other things that are => S {\x.exist(x)} 
#right => ADJ {\x.right(x,'right')}
#right => NN {'right'}
#front => ADJ {\x.front(x)}
#front => NN {'front'}
#behind => ADV {\x.filter_size(x,'behind')}
#behind => NN {'behind'}


##Inference
semantics = True
lex_inference = lexicon.fromstring(r"""
    :- NN, INP, ADJ, DET, IN

    DET :: NN/NN
    ADJ :: NN/NN
    IN :: (NN\NN)/NN
    ADV :: NN/ADJ
    QS :: NN/NN


    the => DET {\P.(P)}
    the => DET {\P.prueba(P)}
    big => ADJ {\x.filter_size(x,'large')}
    object => NN {scene}
   
    """, include_semantics=semantics)

print(lex_inference.__str__())
parser = chart.CCGChartParser(lex_inference, chart.DefaultRuleSet)

sentence = "the purple big object"
new_lex = parser.genlex(sentence) 
print(new_lex.__str__())





 