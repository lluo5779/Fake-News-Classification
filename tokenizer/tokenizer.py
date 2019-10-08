"""
Singleton Tokenizer Class (module)

Enables the transformation of raw data into consumable tokens for models.

The signature for using this class is to import the module as a whole and then
call functions from the module.

:Example: 
from tokenizer import tokenizer
tokenset = tokenizer.tokenize(...)

Public Functions:
:func:`tokenizer.tokenize`

Public Classes:
:class:`tokenizer.TokenSet`
"""

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import os
import pandas as pd
import torch
from pytorch_transformers import BertTokenizer
from typing import Dict, List, Tuple

###
# PRIVATE FUNCTIONS
###


def __build_token_ids(dataframe: pd.DataFrame, **kwarg) -> Dict:
    """Builds a Dict mapping words to indices.

    :param dataframe: raw input CSV as a DataFrame

    """
    words = Counter()
    for tokens in dataframe.Comment:
        words.update(tokens)
    words = sorted(words, key=words.get, reverse=True)
    word2idx = {o: i for i, o in enumerate(words)}
    count = len(list(word2idx.keys()))
    word2idx.update({
        '_PAD': count,
        '_UNK': count + 1,
        '_MASK': count + 2,
        '_END': count + 3
    })
    return word2idx


def __parse_raw_text(text: List, word2idx: Dict, add_end_token: bool,
                     **kwargs) -> np.array:
    """Converts a text str into an array of tokens.
    
    :param text: preprocessed input text for a single venture session
    :param word2idx: Dict mapping words to indices
    :param add_end_token: bool indicating if _END token should be appended

    """
    tokens = []
    for word in text:
        if word not in word2idx:
            raise Exception(f'\'{word}\' not present in word2idx.')
        tokens.append(word2idx[word])
    if add_end_token:
        tokens.append(word2idx['_END'])
    return np.array(tokens, dtype=np.int64)


def __pad_array(size: int, s_ngram: int, s_step: int) -> int:
    """Outputs number of padding tokens to add to token_array.

    Causes the window to compute up to when the window 'falls off'
    the token_array.

    :param size: size of the token_array
    :param s_ngram: size of the ngram
    :param s_step: size of the step

    """
    
    if (size < 1) or (s_ngram < 1) or (s_step < 1):
        raise Exception(f'size ({size}), s_ngram ({s_ngram}), and s_step ({s_step}) must be 1 or greater.')
    
    if s_step > s_ngram:
        raise Exception(f's_step ({s_step}) should not be greater than s_ngram ({s_ngram}).')

    if s_step > size:
        raise Exception(f's_step ({s_step}) should not be greater than size ({size}).')

    if size < s_ngram:
        s_pad = s_ngram - size
    elif (size - s_ngram) % s_step == 0:
        s_pad = 0
    else:
        s_pad = (size - s_ngram)
        s_pad = int(
            np.round(s_step * (1 - ((s_pad / s_step) - (s_pad // s_step)))))
    return s_pad


def __split_tokens(token_array: np.array, s_ngram: int, s_step: int,
                   pad_id: int, **kwargs) -> torch.Tensor:
    """Splits token_array into a 2D Tensor with fixed second dimension size.

    The second dimension should be the number of tokens in the ngram.

    :param token_array: complete array of tokens for a single venture
    :param s_ngram: the size of the ngram
    :param s_step: the step size between ngrams

    If s_ngram == s_step, the token_series will be broken evenly by the size
    of the ngram.
    
    """
    n_pad = __pad_array(token_array.shape[0], s_ngram, s_step)
    if n_pad > 0:
        padded_array = np.concatenate([token_array, np.ones([n_pad]) * pad_id])
    else:
        padded_array = token_array
    t = torch.from_numpy(padded_array)
    return t.unfold(0, s_ngram, s_step)


def __get_max_session_tokens(dataframe: pd.DataFrame) -> int:
    max_session_tokens = dataframe.groupby(['Venture_ID', 'Session']).apply(
                lambda x: len(' '.join(x.Comment.to_list()).split())).max()
    return max_session_tokens


def __get_max_speaker_tokens(dataframe: pd.DataFrame) -> int:
    max_speaker_tokens = dataframe.Comment.apply(lambda x: len(x)).max()
    return max_speaker_tokens


###
# PRIVATE CLASSES
###


class KeyMapper(object):
    """Simplifies identifier handling for data.
    
    Each KeyMapper holds the state for a single identifier type.
    
    """

    def __init__(self, identifier_type: str, ids: Tuple = ()) -> None:
        self.__id_type = identifier_type
        self.__ids = ids
        self.__identifiers = {}
        for id_ in ids:
            self.__identifiers[id_] = None

    def get_data(self) -> Tuple:
        """Gets the raw Tuple of ids."""
        if self.__ids is None:
            self.__ids = ()
        return self.__ids

    def add_data(self, data: torch.Tensor, identifier: int,
                 identifier_type: str) -> None:
        """Adds the indices for the identifier to the map.

        :param data: a Tensor where the first index is the number of data points
        :param identifier: the venture, session, or speaker ID
        :param identifier_type: the string of the type for type checking

        """
        assert self.__id_type == identifier_type, "Types do not match."
        self.__identifiers[identifier] = None
        ids = (identifier, ) * data.shape[0]
        self.__ids = self.get_data() + ids

    def get_map(self) -> Dict:
        """Gets a Dict keyed by identifiers indexing boolean array of indices.

        The boolean array gives all the indices for that identifier.

        :param identifiers: a List of int IDs

        """
        data = np.array(self.get_data())
        datamap = {}
        for identifier in self.__identifiers:
            datamap[identifier] = torch.from_numpy(
                np.where(np.equal(data, identifier))[0])
        return datamap


###
# PUBLIC CLASSES
###


class TokenSet(object):
    """The main class for interacting with the tokenized data.

    The tokenizer singleton has a single function 'tokenize' that returns a
    TokenSet object as its output. Data, data partitions, labels, label
    partitions, and oversampling methods are included in this class.

    :param all_tokens: a Tensor of all token windows for all ventures and sessions
    :param word2idx: a Dict mapping words to their IDs
    :param session_mapper: a KeyMapper object holding session indices
    :param venture_mapper: a KeyMapper object holding venture indices
    :param funding_mapper: a KeyMapper object holding funding indices

    :Example: X, y = tokenset.get_venture_data(), tokenset.get_all_labels()

    :Example: x_res, y_res = tokenset.random_oversample(TokenSet.SAMPLE_CLASS)

    """
    # Resampling types
    SAMPLE_CLASS = 0
    SAMPLE_SESSION = 1
    MAX = '_MAX'

    def __init__(self, all_tokens: torch.Tensor, word2idx: Dict,
                 max_session_tokens: int, max_speaker_tokens: int,
                 add_end_token: bool,
                 session_mapper: KeyMapper, venture_mapper: KeyMapper,
                 funding_mapper: KeyMapper, cohort_mapper: KeyMapper,
                 site_mapper: KeyMapper, speaker_mapper: KeyMapper):
        self.all_tokens = all_tokens
        self.word2idx = word2idx
        self.max_session_tokens = max_session_tokens
        self.max_speaker_tokens = max_speaker_tokens
        self.add_end_token = add_end_token
        self.session_mapper = session_mapper
        self.venture_mapper = venture_mapper
        self.funding_mapper = funding_mapper
        self.cohort_mapper = cohort_mapper
        self.site_mapper = site_mapper
        self.speaker_mapper = speaker_mapper
        self.session_map = session_mapper.get_map()
        self.venture_map = venture_mapper.get_map()
        self.site_map = site_mapper.get_map()
        self.cohort_map = cohort_mapper.get_map()
        self.speaker_map = speaker_mapper.get_map()
        self.counts = None

    def __eq__(self, other): # can't include own class as type
        """Value object equality check based on identical internal structure.

        :param other: the other TokenSet

        """
        bool_check = True
        self_all_data = self.get_all_data().numpy()
        other_all_data = other.get_all_data().numpy()
        
        if self_all_data.shape != other_all_data.shape:
            return False
        
        bool_check &= np.all(np.equal(self_all_data,other_all_data))

        for mapper in ['session_mapper',
                       'venture_mapper',
                       'funding_mapper',
                       'cohort_mapper',
                       'site_mapper',
                       'speaker_mapper']:
            self_mapper = getattr(self,mapper).get_data()
            other_mapper = getattr(other,mapper).get_data()

            if np.array(self_mapper).shape != np.array(other_mapper).shape:
                return False

            bool_check &= np.all(np.equal(self_mapper,other_mapper))
        
        bool_check &= sorted(list(self.word2idx.items())) == \
                      sorted(list(other.word2idx.items()))
        return bool_check

    @staticmethod
    def load(file_name: str):
        """Loads the TokenSet from disk.

        :param file_name: the path and file_name to save to

        """
        with open(file_name, 'rb') as f:
            data = np.load(f, allow_pickle=True)
        return TokenSet(torch.tensor(data[0]), dict(data[1]),
                        data[8], data[9], data[10],
                        KeyMapper('session', data[2]),
                        KeyMapper('ventures', data[3]),
                        KeyMapper('funding', data[4]),
                        KeyMapper('cohort', data[5]),
                        KeyMapper('site', data[6]),
                        KeyMapper('speaker', data[7]))

    def save(self, file_name: str):
        """Saves the TokenSet to disk.

        :param file_name: the path and file_name to save to

        """
        data = np.array([
            self.all_tokens.numpy(),
            np.array(list(self.word2idx.items()),
                     dtype=[('word', 'U32'), ('id', 'i4')]),
            np.array(self.session_mapper.get_data()),
            np.array(self.venture_mapper.get_data()),
            np.array(self.funding_mapper.get_data()),
            np.array(self.cohort_mapper.get_data()),
            np.array(self.site_mapper.get_data()),
            np.array(self.speaker_mapper.get_data()),
            np.array(self.max_session_tokens),
            np.array(self.max_speaker_tokens),
            np.array(self.add_end_token),
        ])
        with open(file_name, 'wb') as f:
            np.save(f, data)

    def __get_all_(self, mapper: KeyMapper):
        """Generic private function to get raw mapper data as Tensor.

        :param mapper: one of session_, venture_, or funding_mapper

        """
        return torch.from_numpy(np.array(mapper.get_data()))

    def get_all_labels(self) -> torch.Tensor:
        """Gets Tensor of labels (funding or not) keyed by position."""
        return self.__get_all_(self.funding_mapper)

    def get_all_sessions(self) -> torch.Tensor:
        """Gets Tensor of session values keyed by position."""
        return self.__get_all_(self.session_mapper)

    def get_all_ventures(self) -> torch.Tensor:
        """Gets Tensor of venture IDs keyed by position."""
        return self.__get_all_(self.venture_mapper)

    def get_all_cohorts(self) -> torch.Tensor:
        """Gets Tensor of cohort values keyed by position."""
        return self.__get_all_(self.cohort_mapper)

    def get_all_sites(self) -> torch.Tensor:
        """Gets Tensor of site values keyed by position."""
        return self.__get_all_(self.site_mapper)

    def get_all_speakers(self) -> torch.Tensor:
        """Gets Tensor of speaker ID's keyed by position."""
        return self.__get_all_(self.speaker_mapper)

    def get_venture_set(self) -> np.array:
        """Gets a sorted array of unique venture ID's in TokenSet."""
        return np.unique(np.array(self.venture_mapper.get_data()))

    def get_speaker_set(self) -> np.array:
        """Gets a sorted array of unique speaker ID's in TokenSet."""
        return np.unique(np.array(self.speaker_mapper.get_data()))

    def get_all_data(self) -> torch.Tensor:
        """Gets data windows for all sessions and ventures."""
        return self.all_tokens

    def get_xy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets train and label data keyed by position."""
        return self.get_all_data(), self.get_all_labels()

    def get_token_count(self) -> int:
        """Gets number of unique words mapped from entire dataset."""
        return len(list(self.word2idx.keys()))

    def get_counts(self, sessions: List) -> Dict:
        """Gets a Dict of counts indexed by venture then session.

        :param sessions: sessions to include

        """
        if self.counts == None:
            self.counts = {TokenSet.MAX: 0.}
            venture_pos = self.get_all_ventures().numpy()
            session_pos = self.get_all_sessions().numpy()
            speaker_pos = self.get_all_speakers().numpy()
            for venture in self.get_venture_set():
                self.counts[venture] = {}
                venture_bool = venture_pos == venture
                for session in sessions:
                    self.counts[venture][session] = {}
                    session_bool = session_pos == session
                    pair_bool = np.logical_and(venture_bool, session_bool)
                    speakers = np.unique(speaker_pos[pair_bool])
                    count_set = []
                    for speaker in speakers:
                        speaker_bool = speaker_pos == speaker
                        count = np.sum(np.logical_and(pair_bool,
                                                      speaker_bool))
                        if count != 0:
                            count_set.append(count)
                            self.counts[venture][session][speaker] = count
                        if self.counts[TokenSet.MAX] < count:
                            self.counts[TokenSet.MAX] = count
        return self.counts

    def __filter(self, indices: np.array, keys: List) -> np.array:
        """Private function to get filtering boolean array.

        :param indices: indices for identifier type
        :param keys: keys to filter on

        .. note:: An empty keys List or a key that is not present in
                  indices list means return all True.

        """
        if keys and set(keys).intersection(indices):
            keep = np.zeros(self.all_tokens.shape[0]) == 1
            for key in keys:
                keep = np.logical_or(keep, indices == key)
        else:
            keep = np.ones(self.all_tokens.shape[0]) == 1
        return keep

    # Typing doesn't work if it returns its own class.
    def filter_data(self,
                    ventures: np.array = None,
                    sessions: np.array = None,
                    labels: np.array = None,
                    cohorts: np.array = None,
                    sites: np.array = None,
                    speakers: np.array = None):
        """Subsets data, labels, and sessions returning new TokenSet.

        :param ventures: List of ventures to include
        :param sessions: List of sessions to include
        :param labels: List of labels to include

        :returns TokenSet

        .. note:: An empty List means include all for all args.

        """
        label_inds = self.get_all_labels().numpy()
        venture_inds = self.get_all_ventures().numpy()
        session_inds = self.get_all_sessions().numpy()
        cohort_inds = self.get_all_cohorts().numpy()
        site_inds = self.get_all_sites().numpy()
        speaker_inds = self.get_all_speakers().numpy()
        label_keep = self.__filter(label_inds, labels)
        venture_keep = self.__filter(venture_inds, ventures)
        session_keep = self.__filter(session_inds, sessions)
        cohort_keep = self.__filter(cohort_inds, cohorts)
        site_keep = self.__filter(site_inds, sites)
        speaker_keep = self.__filter(speaker_inds, speakers)
        keep = np.logical_and.reduce([label_keep,
                                      venture_keep,
                                      session_keep,
                                      cohort_keep,
                                      site_keep,
                                      speaker_keep])
        return TokenSet(self.all_tokens[np.where(keep)], self.word2idx,
                        self.max_session_tokens, self.max_speaker_tokens,
                        self.add_end_token,
                        KeyMapper('session', session_inds[keep]),
                        KeyMapper('ventures', venture_inds[keep]),
                        KeyMapper('funding', label_inds[keep]),
                        KeyMapper('cohort', cohort_inds[keep]),
                        KeyMapper('site', site_inds[keep]),
                        KeyMapper('speaker', speaker_inds[keep]))

    def collapse_speakers(self):
        data = dict(ventures=self.get_all_ventures(),
                    sessions=self.get_all_sessions(),
                    speakers=self.get_all_speakers(),
                    cohorts=self.get_all_cohorts(),
                    sites=self.get_all_sites(),
                    tokens=self.get_all_data().numpy().tolist(),
                    labels=self.get_all_labels())
        df = pd.DataFrame(data)
        aggregate = df.groupby(['cohorts', 'sites', 'ventures', 'sessions']).apply(
                lambda x: np.concatenate(x.tokens.to_list()))
        aggregate = aggregate.rename('tokens').reset_index(0)
        df = df[['ventures', 'sessions', 'cohorts', 'sites', 'labels']]
        df = df.drop_duplicates()
        df = pd.merge(df,
                      aggregate,
                      how='inner',
                      on=['cohorts', 'sites', 'ventures', 'sessions'])
        base = np.ones([self.max_session_tokens+1], dtype=np.int64)
        base *= self.word2idx['_PAD']
        for i, x in enumerate(df.tokens):
            expanded = base.copy()
            values = x[x < self.word2idx['_PAD']]
            try:
                expanded[:values.size] = values
            except:
                raise Exception(f'Cannot collapse speakers: ngram size ({len(data["tokens"][0])}) too small.')
            df['tokens'].iat[i] = expanded
        funding = df.labels.to_numpy()
        all_tokens = np.stack(df.tokens.to_numpy())
        if self.add_end_token:
            pad_inds = np.argmax(all_tokens == self.word2idx['_PAD'],axis=1)
            for i, x in enumerate(pad_inds):
                all_tokens[i][x] = self.word2idx['_END']
        return TokenSet(all_tokens, self.word2idx,
                        self.max_session_tokens, self.max_speaker_tokens,
                        self.add_end_token,
                        KeyMapper('session', df.sessions.to_numpy()),
                        KeyMapper('ventures', df.ventures.to_numpy()),
                        KeyMapper('funding', funding),
                        KeyMapper('cohort', df.cohorts.to_numpy()),
                        KeyMapper('site', df.sites.to_numpy()),
                        KeyMapper('speaker', np.ones_like(funding)))


###
# Public Functions
###


def tokenize(path: str,
             s_ngram: int,
             s_step: int,
             add_end_token: bool,
             df: pd.DataFrame = None,
             word2idx: Dict = None,
             sessions: List = range(1,6),
             **kwargs) -> TokenSet:
    """Tokenizes the raw csv file into the TokenSet object.

    :param csv_file_path: a str to the csv file.
    :param s_ngram: the size of the ngram
    :param s_step: the step size between ngrams
    :param add_end_token: bool indicating if _END token should be appended
    :param df: DataFrame object containing data. If used, path will be ignored.
    :param word2idx: Dict mapping words to indices (optional)
    :param sessions: the sessions to include
    :param kwargs: a collection of key-indexed parameters to change settings
                   in the tokenize process

    .. note:: 
        To save computation time and guarantee consistency word2idx can be
        passed between TokenSets through this param.

    """
    if df == None:
        df = pd.read_csv(path).dropna()
    max_session_tokens = __get_max_session_tokens(df)
    df.Comment = df.Comment.str.split()
    max_speaker_tokens = __get_max_speaker_tokens(df)
    venture_ids = np.unique(df.Venture_ID)
    df = df[df['Session'] != 6] # remove session 6
    if word2idx is None:
        word2idx = __build_token_ids(df)
    session_mapper = KeyMapper('session')
    venture_mapper = KeyMapper('venture')
    funding_mapper = KeyMapper('funding')
    cohort_mapper = KeyMapper('cohort')
    site_mapper = KeyMapper('site')
    speaker_mapper = KeyMapper('speaker')
    all_tokens = []
    for session in sessions:
        session_list = []
        session_df = df[df['Session'] == session]
        if session_df.empty:
            continue
        for venture_id in venture_ids:
            venture_bool = session_df['Venture_ID'] == venture_id
            venture_df = session_df[venture_bool]
            if venture_df.empty:
                continue
            funding = venture_df['Amount_Raised_CAD'].iloc[0]
            if funding < 0:
                raise Exception(f'Venture {venture_id}: funding should never be negative.')
            if funding > 0:
                funding = 1
            else:
                funding = 0 
            for i, row in venture_df.iterrows():
                if not row.Comment and not all(word.isspace() or not word for word in row.Comment):
                    continue
                site = row.Site
                cohort = row.Cohort
                tokens = __parse_raw_text(row.Comment,
                                          word2idx, add_end_token)
                tokens = __split_tokens(tokens, s_ngram, s_step,
                                        word2idx['_PAD'])
                speaker = row.Speaker_ID
                session_list.append(tokens.long())
                venture_mapper.add_data(tokens, venture_id, 'venture')
                funding_mapper.add_data(tokens, funding, 'funding')
                cohort_mapper.add_data(tokens, cohort, 'cohort')
                site_mapper.add_data(tokens, site, 'site')
                speaker_mapper.add_data(tokens, speaker, 'speaker')
        sessions_tensor = torch.cat(session_list)
        session_mapper.add_data(sessions_tensor, session, 'session')
        all_tokens.append(sessions_tensor)
    all_tokens = torch.cat(all_tokens)
    tokenset = TokenSet(all_tokens, word2idx, max_session_tokens,
                        max_speaker_tokens, add_end_token, session_mapper,
                        venture_mapper, funding_mapper, cohort_mapper,
                        site_mapper, speaker_mapper)
    return tokenset


def bert_tokenize(path: str,sessions: List = range(1,6)) -> TokenSet:

    df = pd.read_csv(path).dropna(subset=['Comment'])
    df.Comment = df.Comment.str.split()

    venture_ids = np.unique(df.Venture_ID)
    df = df[df['Session'] != 6] # remove session 6

    session_mapper = KeyMapper('session')
    venture_mapper = KeyMapper('venture')
    funding_mapper = KeyMapper('funding')
    cohort_mapper = KeyMapper('cohort')
    site_mapper = KeyMapper('site')
    speaker_mapper = KeyMapper('speaker')
    all_tokens = []

    bert = BertTokenizer.from_pretrained('bert-base-uncased')
    word2idx = {k:v for k,v in bert.vocab.items()}

    max_tokens = 0
    for session in sessions:
        session_list = []
        session_df = df[df['Session'] == session]
        if session_df.empty:
            continue
        for v_idx, venture_id in enumerate(venture_ids):
            print(f'Tokenizing session {session}, venture {v_idx+1}/{len(venture_ids)}...')
            venture_bool = session_df['Venture_ID'] == venture_id
            venture_df = session_df[venture_bool]
            if venture_df.empty:
                continue
            funding = venture_df['Amount_Raised_CAD'].iloc[0]
            if funding < 0:
                raise Exception(f'Venture {venture_id}: funding should never be negative.')
            if funding > 0:
                funding = 1
            else:
                funding = 0

            for i, row in venture_df.iterrows():
                
                if not row.Comment and not all(word.isspace() or not word for word in row.Comment):
                    continue
 
                comment = '[CLS] ' + ' '.join(row.Comment) + ' [SEP]'
                tokenized_text = bert.tokenize(comment)
                
                max_tokens = max(max_tokens,len(tokenized_text))
                tokens = bert.convert_tokens_to_ids(tokenized_text)

                site = row.Site
                cohort = row.Cohort
                speaker = row.Speaker_ID
                session_list.append(tokens)
               
                tokens = torch.tensor(tokens)
                venture_mapper.add_data(tokens, venture_id, 'venture')
                funding_mapper.add_data(tokens, funding, 'funding')
                cohort_mapper.add_data(tokens, cohort, 'cohort')
                site_mapper.add_data(tokens, site, 'site')
                speaker_mapper.add_data(tokens, speaker, 'speaker')
       
        # add padding to each tensor
        ses_tensors = []
        for tokens in session_list:
            base = np.zeros(512)
            base[:len(tokens)] = tokens
            ses_tensors.append(torch.tensor(base).long())
        
        sessions_tensor = torch.stack(ses_tensors)
        session_mapper.add_data(sessions_tensor, session, 'session')
        all_tokens.append(sessions_tensor)

    # trim padding from all tensors
    trimmed_tokens = []
    for ses_tensor in all_tokens:
        ses_tokens = []
        for t in ses_tensor:
            t = t.numpy()
            ses_tokens.append(torch.tensor(t[:max_tokens]).long())
        trimmed_tokens.append(torch.stack(ses_tokens))

    all_tokens = torch.cat(trimmed_tokens)
    tokenset = TokenSet(all_tokens, word2idx, max_tokens,
                        max_tokens, True, session_mapper,
                        venture_mapper, funding_mapper, cohort_mapper,
                        site_mapper, speaker_mapper)
    return tokenset

def verify_columns(path: str):
    
    df = pd.read_csv(path)

    if 'set' in df.columns:
        df.rename(columns={'set':'Comment'}, inplace=True)
 
    if 'venture_id' in df.columns:
        df.rename(columns={'venture_id':'Venture_ID'}, inplace=True)

    if 'session' in df.columns:
        df.rename(columns={'session':'Session'}, inplace=True)

    if 'Venture_ID' not in df.columns:
        df['Venture_ID'] = 0
 
    if 'Cohort' not in df.columns:
        df['Cohort'] = 0

    if 'Site' not in df.columns:
        df['Site'] = 0

    if 'Speaker_ID' not in df.columns:
        df['Speaker_ID'] = 0

    if 'Amount_Raised_CAD' not in df.columns:
        df['Amount_Raised_CAD'] = 0

    df.to_csv(f'{os.path.splitext(path)[0]}_new.csv')
