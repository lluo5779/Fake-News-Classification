"""
Singleton Parser Class (module)

Enables incremental parsing of the raw text data.

The signature for using this class is to import the module as a whole and then
call functions from the module.

:Example: 
from infrastructure import parsers
new_series = parsers.stage_two_preprocessing(old_series)

Public Functions:
:func:`parsers.stage_one_preprocessing`
:func:`parsers.remove_contractions`
:func:`parsers.stage_two_preprocessing`
:func:`parsers.stage_three_preprocessing`
:func:`parsers.load_data`
:func:`parsers.remove_non_ascii`
:func:`parsers.to_lowercase`
:func:`parsers.underscore_and_slash_to_space`
:func:`parsers.shrink_whitespace`
:func:`parsers.key_ventures`
:func:`parsers.remove_ellipses`
:func:`parsers.remove_punctuation`
:func:`parsers.numbers_to_words`
:func:`parsers.remove_stopwords`
:func:`parsers.remove_contractions`
:func:`parsers.split_text_on_whitespace`
:func:`parsers.to_pos`
:func:`parsers.lemmatize`

"""

import inflect
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
import re
from typing import Dict, List, Tuple

RAW_TXT = './data/processed/datasets/train_whole.csv'
CONTRACTIONS = './data/contractions.pkl'
INDICES = ['id', 'text', 'label']


def load_data(txt_path: str = RAW_TXT) -> pd.DataFrame:
    """Loads raw text and funding data and computes max raised funding.

    :param txt_path: path to raw text file

    """
    df = pd.read_csv(txt_path)[INDICES]
    return df


def split_by_sentences(data: pd.DataFrame) -> pd.DataFrame:
    """

    :param data: a DataFrame of the entire dataset
    """
    
    new_rows = []
    for i,row in data.iterrows():
        if type(row.text) != float:
            line = re.split(r'[.!?]+',row.text)
        curr_rows = []
        for sentence in line:
            if sentence == '':
                continue
            curr_row = row.copy()
            curr_row.text = sentence
            curr_rows.append(curr_row)
        new_rows.extend(curr_rows)

    data_ = pd.DataFrame(new_rows).reset_index(drop=True)
    return data_


def remove_non_ascii(data: pd.Series) -> pd.Series:
    """Removes non-ASCII characters.

    :param data: a Series of Comment data

    """
    return data.replace(r'[^\x00-\x7F]+', value='', regex=True)


def to_lowercase(data: pd.Series) -> pd.Series:
    """Puts all text to lowercase.

    :param data: a Series of Comment data

    """
    return data.str.lower()


def underscore_and_slash_to_space(data: pd.Series) -> pd.Series:
    """Replaces underscores and forward slashes with spaces.

    :param data: a Series of Comment data

    """
    return data.replace(r'[\_/]', value=' ', regex=True)


def shrink_whitespace(data: pd.Series) -> pd.Series:
    """Sets whitespace to a single space.

    :param data: a Series of Comment data

    """
    return data.replace(r'\s+', value=' ', regex=True)



def remove_ellipses(data: pd.Series) -> pd.Series:
    """Removes (...) from text data.

    :param data: a Series of Comment data

    """
    return data.replace(r'\(\.+\)', value='', regex=True)


def remove_punctuation(data: pd.Series) -> pd.Series:
    """Removes punctuation from text data.

    :param data: a Series of Comment data

    """
    return data.replace(r'[^\w\s]+', value='', regex=True)


def numbers_to_words(data: pd.Series) -> pd.Series:
    """Replaces numbers with their matching full text words.

    :param data: a Series of Comment data

    """
    engine = inflect.engine()
    return data.apply(lambda row: re.sub(
        r'\d+', lambda x: engine.number_to_words(x.group()), row))


def remove_stopwords(data: pd.Series) -> pd.Series:
    """Removes stop words from text data.

    :param data: a Series of Comment data

    """
    pattern = r'\b(?:{})\b'.format('|'.join(stopwords.words('english')))
    return data.str.replace(pattern, '')


def remove_contractions(data: pd.Series) -> pd.Series:
    """Replaces contractions with their full form from dictionary.

    :param data: a Series of Comment data

    """
    data_ = data.copy()
    with open(CONTRACTIONS, 'rb') as f:
        contractions = pickle.load(f)
    for kind in contractions.keys():
        words = contractions[kind]
        for word in words:
            word_no_backslash = word.replace('\\','')
            try:
                data_ = data_.str.replace(r'\b{}\b'.format(word), words[word_no_backslash])
            except:
                # for contractions with escape characters
                data_ = data_.str.replace(f'{word_no_backslash}', words[word])
    return data_


def split_text_on_whitespace(data: pd.Series) -> pd.Series:
    """Splits string into text List element-wise in Series.

    :param data: a Series of Comment data

    """
    return data.str.split()


def to_pos(word: List) -> str:
    """Converts a List with a single string to the Part-Of-Speech.

    Convenience function for :func:`parsers.lemmatize`

    :param data: a List with a single string member

    """
    char = pos_tag(word)[0][1][0]
    if char == 'R':
        tag = 'r'
    elif char == 'V':
        tag = 'v'
    elif char == 'J':
        tag = 'a'
    else:
        tag = 'n'
    return tag


def lemmatize(data: pd.Series) -> pd.Series:
    """Lemmatizes individual tokens using WordNet.

    :param data: a Series of Comment data

    """
    lemmatizer = WordNetLemmatizer()
    return data.apply(lambda row: re.sub(
        r'\b\w+\b', lambda match: lemmatizer.lemmatize(
            match.group(), pos=to_pos([match.group()])), row))


def stage_one_preprocessing(data: pd.Series) -> pd.Series:
    """Initial simple text processing.

    Designed to be run before :func:`parsers.remove_contractions`.

    :param data: a Series of Comment data

    """
    data_ = data.dropna()
    print('ascii')
    data_ = remove_non_ascii(data)
    print('lower')
    data_ = to_lowercase(data_)
    print('slash')
    data_ = underscore_and_slash_to_space(data_)
    print('ellipse')
    data_ = remove_ellipses(data_)
    print('white')
    data_ = shrink_whitespace(data_)
    #print('contracts')
    #data_ = remove_contractions(data_)
    return data_


def stage_two_preprocessing(data: pd.Series) -> pd.Series:
    """Second stage of parsing, including major regex replacements.

    Designed to be run after :func:`parsers.remove_contractions`.

    :param data: a Series of Comment data

    """
    # designed to be run after remove_contractions
    data_ = data.dropna()
    data_ = remove_punctuation(data_)
    #data_ = numbers_to_words(data_)
    data_ = remove_stopwords(data_)
    return data_


def stage_three_preprocessing(data: pd.Series) -> pd.Series:
    """Final lemmatizing stage of processing.

    :param data: a Series of Comment data

    """
    data_ = data.dropna()
    data_ = shrink_whitespace(data_)
    data_ = lemmatize(data_)
    return data_

def run_pipeline() -> pd.DataFrame:
    """Main parsing function that runs the entire parsing pipeline
       in sequence.
    
    """

    print('Loading data...')
    data = load_data()
    print('Stage one processing...')
    text = data.text
    text_ = stage_one_preprocessing(text)
    data_ = data.copy()
    data_.text = text_
    print('Splitting by sentences...')
    data_ = split_by_sentences(data_)
    print('Stage two processing...')
    text_ = stage_two_preprocessing(data_.text)
    print('Stage three processing...')
    text_ = stage_three_preprocessing(text_)
    data_.text = text_
    print('Saving file...')
    data_.to_csv(r'./data/stage_three_text.csv')
    return data_


if __name__ == "__main__":
    run_pipeline().to_csv('train_whole_preprocessed.csv')