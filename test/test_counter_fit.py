import sys
import numpy as np
import torch
import random

sys.path.append('../evc_tool/domain/')
from hypothesis import given, assume, settings, Verbosity
from hypothesis import strategies as st
from hypothesis.extra import numpy as exnp

from evc_tool.domain.model import embedding

@settings(verbosity=Verbosity.verbose,deadline=None)
@given(no_components=st.integers(min_value=25,max_value=100),
       glove_epochs=st.integers(min_value=3,max_value=20))
def test_counter_fit(no_components,glove_epochs):
    
    embedding.run('./data/stage_three_plus_contractions_preprocessed_session_notes.csv',
                  336, 1,no_components=no_components,glove_epochs=glove_epochs)


