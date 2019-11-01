from infrastructure import parsers
import pandas as pd
import nltk
import spacy

from nltk import word_tokenize
from nltk import collocations
from nltk import corpus
import collections


def syntacticFeatures(df):

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
            # new_row[label] = row.label
            # print(row)
        # print('new_row: ', new_li)

        return new_li

    def questionRatio(token):
        token = word_tokenize(token)
        question = ['which', 'where', 'why', 'who', 'whose', 'how', 'what', 'when']
        count = 0
        for q in question:
            count += token.count(q)

        if len(token)==0:
            return 0
        else:
            return count/len(token)

    def syntacticRelation(token):
        nlp = spacy.load("en_core_web_sm")

        new_li = []
        prev_id = -1
        dep = {}
        for (i, row) in token.iterrows():
            doc = nlp(str(row.text))
            if prev_id != row.id:
                if i != 0:
                    total = sum(dep.values())
                    for k in dep.keys():
                        dep[k] = dep.get(k)/total
                    new_li.append(dep)
                prev_id = row.id
                dep = {}

            if type(row.text) != type(''):
                continue

            for tok in doc:
                if tok.dep_ not in dep.keys():
                    dep[tok.dep_] = 1
                else:
                    dep[tok.dep_] = dep.get(tok.dep_)+1



    return pd.DataFrame(new_li)['advmod','aux','dobj','amod','ROOT','nsubj','det','compound','pobj','prep']

    def bigramFreq(token):
        token = word_tokenize(token)

        bgm = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(token)

        scored = finder.score_ngrams(bgm.chi_sq)

        prefix_keys = collections.defaultdict(list)
        for key, scores in scored:
            prefix_keys[key[0]].append((key[1], scores))


        for key in prefix_keys:
            prefix_keys[key].sort(key=lambda x: -x[1])

        return prefix_keys



    corpus_tokens = reshapedf(df)

    df_li = []
    i = 0


    syn = syntacticRelation(df)

    for entry in corpus_tokens:
        if (i % 100 == 0):
            print('Starting {}/{} samples.'.format(i, len(corpus_tokens)))

        row = {}
        tokens = list(entry.values())[0]
        row['questionRatio'] = questionRatio(tokens)
        row['bigramFreq'] = bigramFreq(tokens)

        df_li.append(row)
        break
        i += 1

    feature = pd.DataFrame(df_li)

    return pd.merge(feature, syn, right_index=True, left_index=True)

if __name__=='__main__':
    #parsers.run_pipeline()

    filepath = './data/cleaned_compile.csv'
    df = pd.read_csv(filepath)
    syntacticFeatures(df)


