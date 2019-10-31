from collections import Counter
import numpy as np
import zipfile
import string
import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.probability import LidstoneProbDist
from nltk.lm.api import LanguageModel
from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends


def getSyntacticalFeatures(corpus_df):
        
    def reshapedf(raw_df):
        '''
            Input format: df with col id and text at minimum
            Output format: [{id# : listOfTokens}, {id#: listOfTokens}, ...]
        '''
        
        prev_id = -1
        new_li = []
        new_row = {}
        for i, row in raw_df.iterrows():
            if prev_id != row.id: 
                if i != 0:
                    new_li.append(new_row)
                new_row = {}
                prev_id = row.id
                new_row[row.id] = ''
                
            if type(row.text) != type(''):
                continue
            new_row[row.id] = new_row[row.id] + row.text
            #new_row[label] = row.label
            #print(row)
        #print('new_row: ', new_li)
           
        return new_li
        
    def getStandardizedWordEntropy(words) -> int:
        return lm.entropy(words) / len(lm.vocab)
    
    def getMLELM(tokens, n_gram = 2) -> MLE:
        paddedLine = [list(pad_both_ends(tokens, n=n_gram))]
        train, vocab = padded_everygram_pipeline(2, paddedLine)#print(vocab.lookup("in"))
            
        lm = MLE(n_gram)
        lm.fit(train, vocab)
        
        return lm
        
    def getLogFrequencyTags(tokens):
        txt_tagged = nltk.pos_tag(tokens)
        tag_fd = nltk.FreqDist(tag for (word, tag) in txt_tagged)
        tagged_freq = tag_fd.most_common()
        freq = {}
        for (tag, count) in tagged_freq:
            freq[tag] = np.log(count / len(txt_tagged))
            #print(freq[tag])
        return freq
        
    def brunetIndex(tokens):
        total_len = len(tokens)
        vocab_len = len(lm.vocab)
        return total_len / (vocab_len ** -0.165)

   
    def honoreStatistic(tokens):
        vocab_len = len(lm.vocab)
        freq = Counter()
        for word in tokens:
            freq[word] += 1
        words_once = [word for (word, val) in freq.items() if val == 1]
        return np.log(len(tokens)/(1-len(words_once)/len(lm.vocab)))
    
    
    corpus_tokens = reshapedf(corpus_df)
    
    df_li = []
    i = 0
    for entry in corpus_tokens:
        if (i%100 == 0):
            print('Starting {}/{} samples.'.format(i, len(corpus_tokens)))
        
        row = {}
        tokens = list(entry.values())[0]
        #print(tokens)
        lm = getMLELM(tokens)
        row['brunetIndex'] = brunetIndex(tokens)
        row['honoreStatistic'] = honoreStatistic(tokens)
        
        for (key, val) in getLogFrequencyTags(tokens).items():
            row[key+"_logfreq"] = val
        
        df_li.append(row)
        i += 1
        
    return pd.DataFrame(df_li)
        
        
        
if __name__ == "__main__":
    filename = 'data_preprocessed.csv'
    filepath = './data/processed/datasets/'
    
    df = pd.read_csv(filepath+filename, index_col='index')
    #print(df.head())
    a = getSyntacticalFeatures(df)
    print('Writing to CSV')
    a.to_csv('syntacticFeatures_Louis.csv')
    