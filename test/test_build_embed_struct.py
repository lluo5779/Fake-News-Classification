import os, sys
import numpy as np

sys.path.append('../tokenizer')
sys.path.append('../evc_tool/domain/model')
from hypothesis import given, settings, Verbosity
from hypothesis import strategies as st

from tokenizer import tokenizer
from evc_tool.domain.model import embedding


@settings(verbosity=Verbosity.verbose,deadline=35000)
@given(s_ngram=st.integers(min_value=336, max_value=1000),
       no_components=st.integers(min_value=5, max_value=50))
def test_build_embed_struct(s_ngram, no_components):
    """function to test build_embed_struct in evc_tool/domain/model/embedding.py

        :param s_ngram: n_gram size
        :param no_components: number of dimensions per token
    """

    token_set = tokenizer.tokenize(
        'pipeline/data_loader/stage_three_plus_contractions_preprocessed_session_notes.csv',
        s_ngram, 1, True)
    glove_object = embedding.train(token_set, no_components=no_components)
    embed_struct = embedding.build_embed_struct(token_set, glove_object)

    no_sentences = len(token_set.collapse_speakers().all_tokens.tolist())

    # check if the final dimensions of the embed_struct are correct
    assert len(embed_struct) == no_sentences
    assert len(embed_struct[0]) == 2259
    assert len(embed_struct[0][0]) == no_components
