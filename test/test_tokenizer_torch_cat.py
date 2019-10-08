import sys
import random
import torch
import numpy as np
import pandas as pd

sys.path.append('../tokenizer')
from hypothesis import given, assume, settings, Verbosity
from hypothesis import strategies as st
from hypothesis.extra import numpy as exnp
from hypothesis.extra import pandas as expd

from tokenizer import tokenizer

# function to generate one random row of data 
def generateRandomData():
    data = np.random.randint(low=0,high=49,size=random.randint(5,10))
    return data


@settings(verbosity=Verbosity.verbose)
@given(data = exnp.arrays(int,random.randint(5,10),elements=st.integers(min_value=0,max_value=49),
                          fill=None,unique=st.booleans()))
def test_tokenizer_torch_cat(data):
    '''this function tests if torch.cat works as expected'''

    all_tensors = []
    total_rows = 0

    # generate random data and split the tokens into ngrams
    # keep track of number of rows
    for i in range(0,random.randint(0,10)):
        random_data = generateRandomData()
        unfolded = tokenizer.__split_tokens(random_data,5,1,50)
        curr_rows = list(unfolded.size())[0]
        total_rows = total_rows + curr_rows
        all_tensors.append(unfolded)

    # append a final tensor and add to number of rows
    unfolded = tokenizer.__split_tokens(data,5,1,50)
    total_rows = total_rows + list(unfolded.size())[0]
    all_tensors.append(unfolded)
    #assert len(all_tensors) == total_rows
    all_cat = torch.cat(all_tensors)

    # check if the number of rows using torch_cat remains consistent 
    assert list(all_cat.size())[0] == total_rows

