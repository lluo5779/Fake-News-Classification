import sys
import numpy as np
import torch

sys.path.append('../tokenizer')
from hypothesis import given, assume, settings, Verbosity
from hypothesis import strategies as st
from hypothesis.extra import numpy as exnp

from tokenizer import tokenizer
from typing import Dict, List, Tuple

@settings(verbosity=Verbosity.verbose)
@given(t=st.lists(st.text()),add_end_token=st.booleans())
def test_tokenizer_parse_raw_text(t: List, add_end_token: bool):

    # create word2idx dictionary from words in t
    word2idx = {word: i for i, word in enumerate(t)}
    count = len(list(word2idx.keys()))
    word2idx.update({
        '_PAD': count,
        '_UNK': count + 1,
        '_MASK': count + 2,
        '_END': count + 3
    })
    # check if the number of tokens is correct after parsing
    assert len(tokenizer.__parse_raw_text(t,word2idx,add_end_token)) == len(t) + int(add_end_token)

@settings(verbosity=Verbosity.verbose)
@given(token_array=exnp.arrays(int,st.integers(min_value=1,max_value=20),
                               elements=st.integers(min_value=0,max_value=19),
                               fill=None,unique=st.booleans()), 
       s_ngram=st.integers(min_value=1,max_value=10),
       s_step=st.integers(min_value=1,max_value=10))
def test_tokenizer_pad_array_and_split_tokens(token_array: np.array, s_ngram: int, s_step: int):
    
    # assign a pad ID that is larger than all generated values
    pad_id = 20

    assume(s_ngram>=s_step)
    assume(len(token_array)>=s_step)
    
    # call split_tokens and pad_array, and see if the number of pad tokens is correct
    unfolded = tokenizer.__split_tokens(token_array,s_ngram,s_step,pad_id)
    pad = tokenizer.__pad_array(len(token_array),s_ngram,s_step)
    
    #assert len(unfolded.shape) == 2
    assert list(unfolded.size())[0] == (len(token_array)+pad-s_ngram)/s_step + 1

