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

@settings(verbosity=Verbosity.verbose,deadline=1000)
@given(ventures=st.lists(st.integers(min_value=1,max_value=100),max_size=3,unique=True),
       sessions=st.lists(st.integers(min_value=1,max_value=5),max_size=3,unique=True),
       labels=st.lists(st.integers(min_value=0,max_value=1),max_size=1),
       cohorts=st.lists(st.integers(min_value=1,max_value=6),max_size=3,unique=True),
       sites=st.lists(st.integers(min_value=1,max_value=8),max_size=3,unique=True),
       speakers=st.lists(st.integers(min_value=1,max_value=500),max_size=3,unique=True))
def test_filter(ventures,sessions,labels,cohorts,sites,speakers):
    
    # create token set from example data
    token_set = tokenizer.tokenize('data/examples/truncated.csv',5,1,True)

    # get data from each KeyMapper
    venture_data = token_set.venture_mapper.get_data()
    session_data = token_set.session_mapper.get_data()
    funding_data = token_set.funding_mapper.get_data()
    cohort_data = token_set.cohort_mapper.get_data()
    site_data = token_set.site_mapper.get_data()
    speaker_data = token_set.speaker_mapper.get_data()

    ### TEST filter with single input ###
    
    # get number of remaining tokens after we filter
    # for selected input, each independent of the others
    n_ventures = len(venture_data)
    if set(ventures).intersection(venture_data):
        n_ventures = 0
        for v in ventures:
            if v in venture_data:
                n_ventures += venture_data.count(v)
    n_sessions = len(session_data)
    if set(sessions).intersection(session_data):
        n_sessions = 0
        for s in sessions:
            if s in session_data:
                n_sessions += session_data.count(s)
    n_funding = len(funding_data)
    if set(labels).intersection(funding_data):
        n_funding = 0
        for l in labels:
            if l in funding_data:
                n_funding += funding_data.count(l)
    n_cohorts = len(cohort_data)
    if set(cohorts).intersection(cohort_data):
        n_cohorts = 0
        for c in cohorts:
            if c in cohort_data:
                n_cohorts += cohort_data.count(c)
    n_sites = len(site_data)
    if set(sites).intersection(site_data):
        n_sites = 0
        for s in sites:
            if s in site_data:
                n_sites += site_data.count(s)
    n_speakers = len(speaker_data)
    if set(speakers).intersection(speaker_data):
        n_speakers = 0
        for s in speakers:
            if s in speaker_data:
                n_speakers += speaker_data.count(s)

    # check if the numbers calcluated above correspond with what 
    # filter_data returns if called with only a single input parameter
    venture_set = token_set.filter_data(ventures=ventures)
    assert n_ventures == len(venture_set.venture_mapper.get_data()) 
    session_set = token_set.filter_data(sessions=sessions)
    assert n_sessions == len(session_set.session_mapper.get_data())
    funding_set = token_set.filter_data(labels=labels)
    assert n_funding == len(funding_set.funding_mapper.get_data())
    cohort_set = token_set.filter_data(cohorts=cohorts)
    assert n_cohorts == len(cohort_set.cohort_mapper.get_data())
    site_set = token_set.filter_data(sites=sites)
    assert n_sites == len(site_set.site_mapper.get_data())
    speaker_set = token_set.filter_data(speakers=speakers)
    assert n_speakers == len(speaker_set.speaker_mapper.get_data())

    ### TEST filter with multiple inputs ###

    # get number of remaining tokens after we filter
    # for selected input, independently at first...
    venture_bool = np.ones(len(venture_data))
    if set(ventures).intersection(venture_data):
        venture_bool = np.zeros(len(venture_data))
        for v in ventures:
            venture_bool = np.logical_or(venture_bool,[int(x==v) for x in venture_data])
    
    session_bool = np.ones(len(session_data))
    if set(sessions).intersection(session_data):
        session_bool = np.zeros(len(session_data))
        for s in sessions:
            session_bool = np.logical_or(session_bool,[int(x==s) for x in session_data])

    funding_bool = np.ones(len(funding_data))
    if set(labels).intersection(funding_data):
        funding_bool = np.zeros(len(funding_data))
        for l in labels:
            funding_bool = np.logical_or(funding_bool,[int(x==l) for x in funding_data])

    cohort_bool = np.ones(len(cohort_data))
    if set(cohorts).intersection(cohort_data):
        cohort_bool = np.zeros(len(cohort_data))
        for c in cohorts:
            cohort_bool = np.logical_or(cohort_bool,[int(x==c) for x in cohort_data])

    site_bool = np.ones(len(site_data))
    if set(sites).intersection(site_data):
        site_bool = np.zeros(len(site_data))
        for s in sites:
            site_bool = np.logical_or(site_bool,[int(x==s) for x in site_data])

    speaker_bool = np.ones(len(speaker_data))
    if set(speakers).intersection(speaker_data):
        speaker_bool = np.zeros(len(speaker_data))
        for s in speakers:
            speaker_bool = np.logical_or(speaker_bool,[int(x==s) for x in speaker_data])

    # ... now combine all booleans generated above to get final count of remaining data
    n_all = np.logical_and(venture_bool,session_bool)
    n_all = np.logical_and(n_all,funding_bool)
    n_all = np.logical_and(n_all,cohort_bool)
    n_all = np.logical_and(n_all,site_bool)
    n_all = np.logical_and(n_all,speaker_bool)

    count = n_all.tolist().count(1)

    # call filter data with multiple inputs, and check if the token counts calculated
    # above correspond with the counts after running the filter method
    token_set = token_set.filter_data(ventures,sessions,labels,cohorts,sites,speakers)

    assert count == len(token_set.venture_mapper.get_data())
    assert count == len(token_set.session_mapper.get_data())
    assert count == len(token_set.funding_mapper.get_data())
    assert count == len(token_set.cohort_mapper.get_data())
    assert count == len(token_set.site_mapper.get_data())
    assert count == len(token_set.speaker_mapper.get_data())

