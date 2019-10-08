import os, sys
import numpy as np
import math
import torch
import h5py
import types

from glove import Corpus, Glove
from pytorch_transformers import BertTokenizer, BertModel, BertConfig

from tokenizer import tokenizer
from pipeline.counterfitting import counterfitting as cf


def save_embeddings(path: str, word_vectors: dict, word2idx: dict):
    """ Save the embeddings for each word to a file. This will save the word
        followed by the embedding on each line, delimited by spaces.

    :param path: the directory/file where the embeddings will be saved.
    :param word_vectors: a dictionary mapping word to embeddings.
    :param word2idx: a dictionary mapping words to their IDs.
    
    """

    idx2word  = {v:k for k,v in word2idx.items()}

    with open(path, 'wb') as f:
        for idx, vec in word_vectors.items():
            vector = vec
            vector /= math.sqrt((vector**2).sum() + 1e-6)
            vector = vector * 1.0
            str_vector = ' '.join([str(i) for i in vector])
            f.write(f'{idx2word[idx]} {str_vector} \n'.encode())


def load_embeddings(path: str) -> dict:
    """ Load an embedding file into a dictionary.

    :param path: the path to the embedding vectors file.
    
    """

    embed_dict = {}
    with open(path,'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(b' ',1)
            key = line[0].lower()
            embed_dict[key] = np.fromstring(line[1],dtype='float32',sep=' ')
    
    return embed_dict


def save_vocabulary(path: str, words: list):
    """ Save the vocabulary in a file, with one word per line.

    :param path: the directory/file where the vocabulary will be saved.
    :param words: a list of words to be saved.
    
    """

    with open(path, 'wb') as f:
        for word in words:
            f.write(f'{word.lower()} \n'.encode())


def train_glove(token_set: tokenizer.TokenSet,
                window: int = 20,
                no_components: int = 25,
                learning_rate: float = 0.05,
                glove_epochs: int = 3,
                no_threads: int = 4,
                verbose: bool = True) -> Glove:
    """Train the TokenSet data to create GloVe embeddings for each token.
    
    :param token_set: a TokenSet object filled with preprocessed token data.
    :param no_components: number of dimensions for each embedding.
    :param learning_rate: the learning rate for the model.
    :param glove_epochs: the number of training iterations.
    :param no_threads: number of treads to use for processing.
    :param verbose: toggle extra print output.
    
    """

    # get a list of sentences per venture/session pair
    token_set = token_set.collapse_speakers()
    sentences = token_set.all_tokens.tolist()

    # add data to corpus
    corpus = Corpus(dictionary=token_set.word2idx)
    print('Fitting Corpus...')
    corpus.fit(sentences, window, ignore_missing=True)

    # train the data
    glove_object = Glove(no_components, learning_rate)
    print('Fitting Glove Vectors...')
    glove_object.fit(corpus.matrix,
                     epochs=glove_epochs,
                     no_threads=no_threads,
                     verbose=verbose)
    glove_object.add_dictionary(corpus.dictionary)

    return glove_object


def build_embed_struct(sentences: np.array, word_vectors: dict, 
                       word2idx: dict = None) -> np.array:
    """ Build a 3D embedding structure given an array of sentences/tokens.
    The structure will be of size n_sentences x n_tokens x n_components.

    :param sentences: a 2D array of sentences/tokens.
    :param word_vectors: a dictionary mapping words to embeddings.
    :param word2idx: dictionary mapping words to ids (optional).

    """

    idx2word = None
    if word2idx:
        idx2word = {v:k for k,v in word2idx.items()}

    # replace each token ID in sentences with the corresponding embedding
    embed_struct = []
    for i, sentence in enumerate(sentences):
        print(f'Building sentence {i}/{len(sentences)}...')
        embed_struct.append([])
        for word in sentence:
            if idx2word:
                word = idx2word[word].lower().encode()
            embed_struct[i].append(word_vectors[word])

    return np.array(embed_struct)


def counter_fit(config_path: str = './pipeline/bash_files/experiment_parameters.cfg') -> dict:
    """ Run the counterfitting procedure and return the counter-fitted embeddings.

    :param config_path: The path to the configuration file.
    
    """

    cf.run_experiment(config_path)
    embed_dict = load_embeddings('./pipeline/bash_files/results/counter_fitted_vectors.txt')
    
    return embed_dict


def run_pipeline(path: str,
                 s_ngram: int,
                 s_step: int,
                 window: int = 20,
                 no_components: int = 100,
                 learning_rate: float = 0.05,
                 glove_epochs: int = 10,
                 no_threads: int = 4,
                 verbose: bool = True) -> np.array:
    """ Run the entire pipeline:
    
    1. tokenize parsed data.
    2. build embeddings.
    3. counter-fit embeddings.
    4. return a 3d embedding structure.

    :param path: the path to the preprocessed session notes.
    :param s_ngram: the size of the ngrams.
    :param s_step: the size of steps to skip between each ngram.
    :param no_components: number of dimensions for each embedding.
    :param learning_rate: the learning rate for the model.
    :param glove_epochs: the number of training iterations.
    :param no_threads: number of treads to use for processing.
    :param verbose: toggle extra print output.
    
    """

    # tokenize the data
    token_set = tokenizer.tokenize(path, s_ngram, s_step, True)
    
    # train the glove object and create embeddings for each word
    glove_object = train_glove(token_set,
                               window=window,
                               no_components=no_components,
                               learning_rate=learning_rate,
                               glove_epochs=glove_epochs,
                               no_threads=no_threads,
                               verbose=verbose)

    # save the embeddings and vocabulary for counter-fitting
    save_embeddings('./pipeline/bash_files/word_vectors/vectors.txt', 
                    glove_object.word_vectors, token_set.word2idx)
    save_vocabulary('./pipeline/bash_files/word_vectors/vocabulary.txt',
                    [word for word in token_set.word2idx.keys()])
    
    # perform counter-fitting procedure
    cf_embeddings = counter_fit()

    # get a list of sentences per venture/session pair
    token_set = token_set.collapse_speakers()
    sentences = np.array(token_set.all_tokens.tolist())

    # build the final embedding structure
    embed_struct = build_embed_struct(sentences,cf_embeddings)

    return embed_struct


###
# BERT functions
###


def build_bert_embeddings(token_set: tokenizer.TokenSet,
                          embed_path: str = './data/embeddings.h5',
                          wordi_path: str = './data/word_ids.h5') -> tuple:
    """ Create BERT embeddings from a TokenSet and save to an h5 file.

    :param token_set: a TokenSet object filled with BERT tokenized data.
    :param embed_path: the directory/file where the embeddings will be saved.
    :param wordi_path: the directory/file where the word_ids will be saved.

    """
    
    # get a list of all sentences
    sentences = token_set.all_tokens

    # load the pre-trained BERT model
    #config = BertConfig.from_pretrained('bert-base-uncased')
    #config.output_hidden_states = True
    #model = BertModel(config)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval() # put model in evaluation mode
    #model.to('cuda')

    # initialize empty h5py files
    hf_e = h5py.File(embed_path,'w')
    hf_w = h5py.File(wordi_path,'w')

    # initialize dataset for embeddings and word_ids
    embeddings = hf_e.create_dataset('embeddings', (len(sentences),len(sentences[0]),768),dtype='float64')
    word_ids = hf_w.create_dataset('word_ids', (len(sentences),len(sentences[0])),dtype='int32')

    # extract embeddings for each token of each sentence
    for i,sentence in enumerate(sentences):
        print(f'Creating embedding {i+1}/{len(sentences)}...')

        segments_ids = np.ones(len(sentence))
        segments_tensor = torch.tensor([segments_ids]).to(torch.int64)#.to('cuda')
        
        with torch.no_grad():
            sentence = sentence.unsqueeze(0).to(torch.int64)#.to('cuda')
            output,_ = model(sentence,segments_tensor)

        for token_i,token in enumerate(sentence[0]):

            word_ids[i,token_i] = int(token.numpy())
            hf_w.flush()

            vec = output[0,token_i]

            # normalize the word vectors
            #vec /= math.sqrt((vec**2).sum() + 1e-6)
            #vec = vec * 1.0
            #vec = vec.numpy().tolist() 

            embeddings[i,token_i,:] = vec
            hf_e.flush()
 
    hf_e.close()
    hf_w.close()

    return load_bert_embeddings(embed_path,wordi_path)


def load_bert_embeddings(embed_path: str = './data/embeddings.h5', 
                         wordi_path: str = './data/word_ids.h5') -> tuple:
    """ Load BERT embeddings for each sentence/word from h5 file in path.

    
    :param embed_path: the directory/file where the embeddings are saved.
    :param wordi_path: the directory/file where the word_ids are saved.

    """

    hf_e = h5py.File(embed_path,'r')
    hf_w = h5py.File(wordi_path,'r')

    embeddings = hf_e['embeddings']
    word_ids = hf_w['word_ids']

    return (embeddings,word_ids)


def collapse_bert_embeddings(embeddings: np.array = None,
                             word_ids: np.array = None,
                             embed_path: str = './data/embeddings.h5',
                             wordi_path: str = './data/word_ids.h5',
                             func: types.FunctionType = lambda x: np.mean(x,axis=0)) -> dict:
    """ Collapse BERT embeddings so that there is only one unique embedding per word.
    Embeddings can be collapsed using a specified function (default is mean).
    
    :param word2idx: provide a dictionary mapping words to ids.
    :param embeddings: a numpy array of a BERT embeddings structure (optional).
    :param word_ids: a numpy array of the word_ids for each sentence (optional).
    :param embed_path: the directory/file where the embeddings are saved.
    :param wordi_path: the directory/file where the word_ids are saved.
    :param func: the specified function used to collapse the embeddings.
    
    """

    if not embeddings and not word_ids:
        embeddings, word_ids = load_bert_embeddings(embed_path,wordi_path)

    grouped_embeddings = {}
    for si, sentence in enumerate(word_ids):
        print(f'Collapsing sentence {si+1}/{len(word_ids)}...')
        for ti, token in enumerate(sentence):
            if token in grouped_embeddings:
                np.append(grouped_embeddings[token],embeddings[si][ti])
            else:
                grouped_embeddings[token] = np.array([embeddings[si][ti]])

    collapsed = {k: func(v) for k, v in grouped_embeddings.items()}

    return collapsed


def run_bert_pipeline(path: str = './data/stage_three_session_notes.csv',
                      embed_path: str = './data/embeddings.h5',
                      wordi_path: str = './data/word_ids.h5',
                      collapse = True) -> tuple:
    """ Run the entire BERT pipeline:
    
    1. tokenize parsed data.
    2. build embeddings.
    3. collapse embeddings.
    4. counter-fit embeddings.
    5. return a 3d embedding structure.
    
    :param path: the directory/file containing the source data.
    :param embed_path: the directory/file where the embeddings will be saved.
    :param wordi_path: the directory/file where the word_ids will be saved.
    :param collapse: boolean to collapse to a single embedding for each word.

    """
    
    # tokenize the data
    token_set = tokenizer.bert_tokenize(path)

    # create embeddings for each word in each sentence
    embeddings, word_ids = build_bert_embeddings(token_set,embed_path,wordi_path)
    
    embed_struct = embeddings

    if collapse:
        # collapse the BERT embeddings so that there is only one per word
        collapsed = collapse_bert_embeddings(embeddings,word_ids)

        # save the embeddings and vocabulary for counter-fitting
        save_embeddings('./pipeline/bash_files/word_vectors/vectors.txt',
                        collapsed, token_set.word2idx)
        save_vocabulary('./pipeline/bash_files/word_vectors/vocabulary.txt',
                        [word for word in token_set.word2idx.keys()])

        # perform counter-fitting procedure
        cf_embeddings = counter_fit()

        # build the final embedding structure
        embed_struct = build_embed_struct(word_ids,cf_embeddings,token_set.word2idx)

    return (embed_struct,embeddings,word_ids,cf_embeddings,token_set.word2idx)


