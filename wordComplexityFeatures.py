from collections import Counter
import numpy as np
import zipfile
import string
import pandas as pd

import textstat
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

PRONOUNS = ["I", "me", "mine", "my", "you", "yours", "your", "we", "us", "our", "ours"]
QUESTION = ['which', 'where', 'why', 'who', 'whose', 'how', 'what', 'when']


class TextFeatures(object):
    """ An object that extracts features from cleaned text files

    Example: (run this in .git directory)
    from pipeline.feature_extraction import *
    df = pd.read_csv("data/processed/datasets/data_preprocessed.csv")
    feats = TextFeatures(df)
    final_df = feats.get_all_features()

    Attributes:
    """

    def __init__(self,
                 df=None):
        self.lms = {}
        self.df = df

    def reshapedf(self, raw_df=None):
        '''
            Returns df that joins all sentences in an article. Each row of
            resulting df is an article.

            Input format: df with col names id and text at minimum
            Output format: [{id#1 : listOfArticle1Tokens}, {id#2: listOfArticle2Tokens}, ...]
        '''

        if raw_df is None:
            raw_df = self.df
        raw_df = raw_df.dropna()

        df = pd.DataFrame(columns=['text', 'label'])
        df['text'] = raw_df.groupby(['id'])['text'].apply("".join)
        df['label'] = raw_df.groupby(['id'])['label'].first()

        return df

    def getWordComplexityScore(self, tokens, i):
        # A higher score means a document takes a higher education level to read
        if (i == 1):
            score = textstat.gunning_fog(tokens)
        elif (i == 2):
            # Texts of fewer than 30 sentences are statistically invalid, because the SMOG formula was normed on 30-sentence samples.
            # textstat requires atleast 3 sentences per article for a result.
            score = textstat.smog_index(tokens)
        else:
            score = textstat.flesch_kincaid_grade(tokens)

        return score


    def get_complexity_features(self, df=None):
        if df is None:
            df = self.df
        df_li = pd.DataFrame()

        df_li['wordComplexityScore_GunningFog'] = df['text'].apply(lambda x: self.getWordComplexityScore(x, 1))
        df_li['wordComplexityScore_SMOG'] = df['text'].apply(lambda x: self.getWordComplexityScore(x,2))
        df_li['wordComplexityScore_FleschKincaid'] = df['text'].apply(lambda x: self.getWordComplexityScore(x, 3))
        return df_li

    def get_all_features(self, df=None):
        if df is None:
            df = self.df

        print("Getting complexity features...")
        lexical = self.get_complexity_features()
        lexical['id'] = lexical.index

        return lexical
