import sys
import numpy as np
import torch
import random

sys.path.append('../tokenizer')
from hypothesis import given, assume, settings, Verbosity
from hypothesis import strategies as st
from hypothesis.extra import pandas as expd

from infrastructure import parsers
from typing import Dict, List, Tuple

#def insert_random_keys(text):
#
#    max_iter = random.randint(0,3)
#
#    split_at = []
#    while len(split_at) < max_iter:
#        x = random.randint(0,len(text))
#        if x not in split_at:
#            split_at.append(x)
#
#    split_at = split_at.sort()
#    if 0 not in split_at:
#        split_at = [0].extend(split_at)
#
#    new_text = []
#    for i in split_at[1:]:
#        new_text.append(text[i-1:i])
#
#    new_text.append(text[i[-1]:len(text)])


@settings(verbosity=Verbosity.verbose)
@given(data = expd.series(elements=st.text(u'abcdefghijklmnopqrstuvwxyz0123456789-_/ ',min_size=25,max_size=500),
                          dtype=None,index=None,fill=None,unique=True),
       upper = expd.series(elements=st.text(u'ABCDEFGHIJKLMNOPQRSTUVWXYZ',min_size=25,max_size=500),
                          dtype=None,index=None,fill=None,unique=True),
       non_ascii = expd.series(elements=st.text(st.characters(min_codepoint=256,max_codepoint=1000),min_size=25,max_size=500),
                          dtype=None,index=None,fill=None,unique=True),
       numbers = expd.series(elements=st.text(u'0123456789',min_size=1,max_size=10),
                          dtype=None,index=None,fill=None,unique=True),
       data_k_e = expd.series(elements=st.text(u'abcdefghijklmnopqrstuvwxyz0123456789-_/ ',min_size=25,max_size=500).map(
                          insert_random_keys).example(),dtype=None,index=None,fill=None,unique=True))
def test_parsers(data,non_ascii,upper,numbers):

    # TEST remove_non_ascii
    ascii_only = parsers.remove_non_ascii(non_ascii)

    for row in ascii_only:
        assert row == ''

    # TEST to_lowercase
    lower = parsers.to_lowercase(upper)

    for row in lower:
        assert row.islower()
    
    # TEST underscore_and_slash_to_space
    new_data = parsers.underscore_and_slash_to_space(data)

    for row in new_data:
        assert '_' not in row
        assert '/' not in row

    # TEST shrink_whitespace

    new_data = parsers.shrink_whitespace(data)

    # make sure there isn't more than one whitespace in a row
    for row in new_data:
        whitespace = False
        for char in row:
            if char.isspace():
                assert not whitespace
                whitespace = True
            else:
                whitespace = False

    # TEST key_speakers

    

    # TEST remove_ellipses

    

    # TEST remove_punctuation

    # TEST numbers_to_words
    letters = parsers.numbers_to_words(numbers)

    for row in letters:
        assert not any(char.isdigit() for char in row)

    # TEST remove_stopwords

    # TEST remove_contractions

    # TEST split_text_on_whitespace

    # TEST to_pos

    # TEST lemmatize
