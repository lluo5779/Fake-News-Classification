from collections import Counter
import numpy as np
import pandas as pd

import nltk

from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.text import Text
import re


def getLexicalFeatures(corpus_df):
    sentenceDictionary={}
    minSentenceDictionary={}
    punctuationDictionary={}

    def getAndRemovePunc(raw_df):
        i=0
        prev_id = -1
        new_li = []
        new_row = {}
        punc = 0
        nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
        for i, row in raw_df.iterrows():
            if prev_id != row.id:
                if i != 0:
                    for w in new_row[row.id-1]:
                        if nonPunct.match(w):
                            punc=punc
                        else:
                            if w != ' ':
                                punc+=1
                    new_li.append(new_row)
                    punctuationDictionary[row.id-1] = punc
                    punc=0
                new_row = {}
                prev_id = row.id
                new_row[row.id] = ''
            if type(row.text) != type(''):
                continue
            new_row[row.id] = new_row[row.id] + row.text

        return raw_df

    def reshapedf(raw_df):
        '''
            Input format: df with col id and text at minimum
            Output format: [{id# : listOfTokens}, {id#: listOfTokens}, ...]
        '''

        prev_id = -1
        new_li = []
        new_row = {}
        sentenceCount = 0
        sentenceLength = 0
        for i, row in raw_df.iterrows():
            sentenceCount += 1
            if prev_id != row.id:
                if i != 0:
                    new_li.append(new_row)
                    sentenceDictionary[row.id-1] = sentenceCount-1
                    sentenceCount = 0
                new_row = {}
                prev_id = row.id
                new_row[row.id] = ''
            if type(row.text) != type(''):
                continue
            sentenceLength = len(row.text)
            if (row.id in minSentenceDictionary.keys()):
                if (sentenceLength < minSentenceDictionary[row.id]):
                    minSentenceDictionary[row.id] = sentenceLength
            else:
                minSentenceDictionary[row.id] = sentenceLength
            new_row[row.id] = new_row[row.id] + row.text
            # new_row[label] = row.label
            # print(row)
        # print('new_row: ', new_li)

        return new_li

    def getWordCount(tokens):
        tokenizer = RegexpTokenizer(r'\w+')
        zen_no_punc = tokenizer.tokenize(tokens)
        return len(zen_no_punc)

    def getDistinctWords(tokens):
        tokenizer = RegexpTokenizer(r'\w+')
        zen_no_punc = tokenizer.tokenize(tokens)
        return len(set(w.title() for w in zen_no_punc if w.lower() not in stopwords.words()))

    def getAverageWordLength(tokens):
    #    Total characters / Total words
        tokenizer = RegexpTokenizer(r'\w+')
        zen_no_punc = tokenizer.tokenize(tokens)
        char=0
        for word in zen_no_punc:
            char+=len(word)
        return float(char/len(zen_no_punc))

    def getAvgSentenceLength(tokens,article):
        # In # of words
        tokenizer = RegexpTokenizer(r'\w+')
        zen_no_punc = tokenizer.tokenize(tokens)
        sentences = sentenceDictionary[article]
        return (float(len(zen_no_punc)/sentences))

    #Just to remove punctuation
    corpus_df = getAndRemovePunc(corpus_df)

    corpus_tokens = reshapedf(corpus_df)

    df_li = []
    i = 0
    for entry in corpus_tokens:
        if (i % 100 == 0):
            print('Starting {}/{} samples.'.format(i, len(corpus_tokens)))

        row = {}
        tokens = list(entry.values())[0]
        # print
        row['wordCount'] = getWordCount(tokens)
        row['distinctWords'] = getDistinctWords(tokens)
        row['avgWordLength'] = getAverageWordLength(tokens)
        row['typeTokenRatio'] = float(getDistinctWords(tokens)/getWordCount(tokens))
        row['avgSentenceLength'] = getAvgSentenceLength(tokens , i)
        row['minSentenceLength'] = minSentenceDictionary[i]
        row['punctuationCount'] = punctuationDictionary[i]

        df_li.append(row)
        i += 1

    return pd.DataFrame(df_li)


if __name__ == "__main__":
    filename = 'data_preprocessed.csv'
    filepath = 'C:/Users/Overwatch/Python/prediction_tool/data/processed/datasets/'

    df = pd.read_csv(filepath + filename) #, index_col='index'
    # print(df.head())
    a = getLexicalFeatures(df)
    print('Writing to CSV')
    a.to_csv('lexicalFeatures_Jennee.csv')
