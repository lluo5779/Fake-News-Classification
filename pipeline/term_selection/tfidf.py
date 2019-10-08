import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import csv

path = '/home/cdl-ai/Documents/CDL_AI_Models/Kyndi-CDL-AI/GloVe/Data'
token_dict = {}
file_dict = {}

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

for dirpath, dirs, files in os.walk(path):
    for f in files:
        fname = os.path.join(dirpath, f)
        print("fname=", fname)
        with open(fname) as pearl:
            text = pearl.read()
            token_dict[f] = text.lower().translate(string.punctuation)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
feature_names = tfidf.get_feature_names()


evc_dict = {}
with open("/home/cdl-ai/Documents/CDL_AI_Models/Kyndi-CDL-AI/Data/evc.csv") as csvfile:
	reader = csv.reader(csvfile, delimiter = '\t')
	for row in reader:
		if row[6] != "Total Funding CAD" and row[6] != "(empty)" and row[6] != "N/A":
			if row[6] != str(0):
				evc_dict[row[1]] = 1
			else:
				evc_dict[row[1]] = 0

for key in token_dict:
	file_dict[key] = tokenize(token_dict[key])

