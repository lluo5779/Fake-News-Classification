import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

sys.path.append('../evc_tool/domain/')
from evc_tool.domain.model import embedding

def compare_counter_fit(word1,word2):
    """ Generates plots to test the optimal number of epochs and components for creating
        embeddings and counter-fitting. This should be run after test_counter_fit.py.

        :param word1: the first word in the word pair to compare.
        :param word2: the second word in the word pair to compare.
    """

    trials = sorted([x[1] for x in os.walk('./pipeline/bash_files/word_vectors')][0])
    result_trials = [x[1] for x in os.walk('./pipeline/bash_files/results')][0]
    vocabulary = []

    with open('./pipeline/bash_files/word_vectors/vocabulary.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            vocabulary.append(line.rstrip())

    ind1 = vocabulary.index(word1)
    ind2 = vocabulary.index(word2)

    for i,trial in enumerate(trials):

        if trial not in result_trials:
            print(str(i+1)+'. Skipping '+trial+'...')
            continue

        print(str(i+1)+'. Plotting '+trial+'...')
        t = trial.split('_')
        epochs = int(''.join([c for c in t[0] if c.isdigit()]))
        components = int(''.join([c for c in t[1] if c.isdigit()]))

        pre_cf_data = embedding.load_embeddings(f'./pipeline/bash_files/word_vectors/{trial}/vectors.txt')
        post_cf_data = embedding.load_embeddings(f'./pipeline/bash_files/results/{trial}/counter_fitted_vectors.txt')

        pre_dist = euclidean_distances([v for v in pre_cf_data.values()])
        post_dist = euclidean_distances([v for v in post_cf_data.values()])
        dist_diff = np.subtract(post_dist,pre_dist)

        color = 'k'
        if dist_diff[ind1][ind2] < 0:
            color = 'r'

        plt.scatter(epochs,components,c=color,s=dist_diff[ind1][ind2]**2*500)

    plt.xlabel('# Epochs')
    plt.ylabel('# Components')
    plt.title('Change in Distance')

    plt.show()
    plt.savefig(f'./pipeline/bash_files/results/euclidean_{word1}_vs_{word2}.png')

if __name__=='__main__':

    word1, word2 = sys.argv[1:]
    compare_counter_fit(word1,word2)
