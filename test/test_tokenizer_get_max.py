import sys
import numpy as np
import pandas as pd

sys.path.append('../tokenizer')
from hypothesis import given, assume, settings, Verbosity
from hypothesis import strategies as st
from hypothesis.extra import numpy as exnp
from hypothesis.extra import pandas as expd

from tokenizer import tokenizer

@settings(verbosity=Verbosity.verbose)
@given(expd.data_frames([expd.column('Venture_ID',elements=st.integers(min_value=0,max_value=100),dtype=int),
                         expd.column('Session',elements=st.integers(min_value=1,max_value=6),dtype=int),
                         expd.column('Comment',elements=st.text(u'abcdefghijklmnopqrstuvwxyz0123456789-_ ',
                                     min_size=25,max_size=500),dtype=str,fill=None,unique=True)]))
def test_tokenizer_get_max_session_tokens(dataframe: pd.DataFrame):

    assume(dataframe.empty==False)

    # get list of all venture IDs, session IDs, and comments
    venture_ids = dataframe.groupby(['Venture_ID','Session'])['Venture_ID'].apply(pd.Series).tolist()
    session_ids = dataframe.groupby(['Venture_ID','Session'])['Session'].apply(pd.Series).tolist()
    comments = dataframe.groupby(['Venture_ID','Session'])['Comment'].apply(pd.Series).tolist()

    ### treats each row independently: will generate falsifying examples whenever combined
    ### word/token count for a venture/session pair is larger than any single row of words/tokens
    #comment_count = [len(x.split()) for x in comments]
    #assert tokenizer.__get_max_session_tokens(dataframe) == max(comment_count)

    # Count length of each row of tokens. Counts for the same 
    # venture and session is accumulated
    count = {}
    for i, comment in enumerate(comments):
        combined_id = str(venture_ids[i])+'_'+str(session_ids[i])
        
        if combined_id not in count:
            count[combined_id] = len(comment.split())
        else:
            count[combined_id] = count[combined_id] + len(comment.split())

    # check if the max count from above corresponds with what __get_max_session_tokens returns
    assert tokenizer.__get_max_session_tokens(dataframe) == max(list(count.values()))


@settings(verbosity=Verbosity.verbose)
@given(expd.data_frames([expd.column('Speaker_ID',elements=st.integers(min_value=0,max_value=100),dtype=int),
                         expd.column('Session',elements=st.integers(min_value=1,max_value=6),dtype=int),
                         expd.column('Comment',elements=st.text(u'abcdefghijklmnopqrstuvwxyz0123456789-_ ',
                                     min_size=25,max_size=500),dtype=str,fill=None,unique=True)]))
def test_tokenizer_get_max_speaker_tokens(dataframe: pd.DataFrame):

    assume(dataframe.empty==False)

    # get list of each row of comments and get the length of each row
    comments = dataframe.groupby(['Speaker_ID','Session'])['Comment'].apply(pd.Series).tolist()
    count = [len(x.split()) for x in comments]

    # change data type of the Comment column from string to list
    dataframe.Comment = dataframe.Comment.str.split()

    # check if the max count from above corresponds with what __get_max_speaker_tokens returns
    assert tokenizer.__get_max_speaker_tokens(dataframe) == max(count)
