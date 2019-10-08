import sys
sys.path.append('../tokenizer')
from tokenizer import tokenizer

def test_tokenizer_tokenSet_eq():

    token_setA = tokenizer.tokenize('data/examples/exampleA.csv',5,1,True)
    token_setB = tokenizer.tokenize('data/examples/exampleB.csv',5,1,True)
    
    token_setC = tokenizer.tokenize('data/examples/exampleC.csv',5,1,True)
    token_setD = tokenizer.tokenize('data/examples/exampleD.csv',5,1,True)
    token_setE = tokenizer.tokenize('data/examples/exampleE.csv',5,1,True)
    token_setF = tokenizer.tokenize('data/examples/exampleF.csv',5,1,True)
    token_setG = tokenizer.tokenize('data/examples/exampleG.csv',5,1,True)
    token_setH = tokenizer.tokenize('data/examples/exampleH.csv',5,1,True)
    token_setI = tokenizer.tokenize('data/examples/exampleI.csv',5,1,True)
    
    token_setJ = tokenizer.tokenize('data/examples/exampleA.csv',3,1,True)
    token_setK = tokenizer.tokenize('data/examples/exampleA.csv',5,2,True)
    token_setL = tokenizer.tokenize('data/examples/exampleA.csv',5,1,False)

    assert token_setA == token_setB # identical
    assert token_setA != token_setC # different cohort
    assert token_setA != token_setD # different site
    assert token_setA != token_setE # different session
    assert token_setA != token_setF # different venture ID
    assert token_setA != token_setG # different speaker ID
    assert token_setA != token_setH # different comment
    assert token_setA != token_setI # one row different 
    assert token_setA != token_setJ # different ngram size
    assert token_setA != token_setK # different step size
    assert token_setA != token_setL # end token vs no end token
