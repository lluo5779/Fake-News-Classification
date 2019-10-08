import sys
import numpy as np
import torch
import random

sys.path.append('../tokenizer')
from hypothesis import given, assume, settings, Verbosity
from hypothesis import strategies as st
from hypothesis.extra import numpy as exnp

from tokenizer import tokenizer
from typing import Dict, List, Tuple

# function that generates a random array of values of size range(1:8) x ngram
def generateRandomData():
    data = np.random.randint(low=0,high=49,size=5)
    for i in range(0,random.randint(0,8)):
        data = np.vstack((data,np.random.randint(low=0,high=49, size=5)))
    return data

@settings(verbosity=Verbosity.verbose)
@given(data = exnp.arrays(int,(random.randint(1,9),5),
                          elements=st.integers(min_value=0,max_value=49),
                          fill=None,unique=st.booleans()),
       identifier=st.integers(min_value=0,max_value=49))
def test_mapper(data,identifier):
    
    # create a speaker mapper
    mapper = tokenizer.KeyMapper('speaker')

    # add some dummy data and record total row count
    row_count = int(data.shape[0])
    for i in range(0,random.randint(0,10)):
        random_data = generateRandomData()
        row_count = row_count + int(random_data.shape[0])
        mapper.add_data(torch.from_numpy(random_data),random.randint(0,49),'speaker')
    
    # check if row count stays consistent after we add random data
    mapper.add_data(torch.from_numpy(data),identifier,'speaker')
    assert len(mapper.get_data()) == row_count


# quick test to check if generateRandomData() works
#if __name__=='__main__':
#
#    data = generateRandomData()
#    print(data)
