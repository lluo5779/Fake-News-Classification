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
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

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
				 df = None):
		self.lms = {}
		self.df = df

	def reshapedf(self, raw_df = None):
		'''
			Returns df that joins all sentences in an article. Each row of 
			resulting df is an article.
			
			Input format: df with col names id and text at minimum
			Output format: [{id#1 : listOfArticle1Tokens}, {id#2: listOfArticle2Tokens}, ...]
		'''
		
		if raw_df is None:
			raw_df = self.df
		raw_df = raw_df.dropna()

		df = pd.DataFrame(columns = [['text', 'label']])
		df['text'] = raw_df.groupby(['id'])['text'].apply("".join)
		df['label'] = raw_df.groupby(['id'])['label'].first()

		return df
		
	def get_standardized_word_entropy(self, words) -> int:
		lm = self.get_MLELM(words)
		return lm.entropy(words) / len(lm.vocab)
	
	def get_MLELM(self, tokens, n_gram = 2) -> MLE:
		'''
			Trains lm and stores in class upon training to be reused
		'''
		paddedLine = [list(pad_both_ends(tokens, n=n_gram))]
		train, vocab = padded_everygram_pipeline(2, paddedLine)
			
		if (tokens not in self.lms.keys()):
			lm = MLE(n_gram)
			lm.fit(train, vocab)
			self.lms[tokens] = lm
			
		return self.lms[tokens]
		
	def get_sqrt_frequency_tags_per_article(self, tokens):
		txt_tagged = nltk.pos_tag(tokens)
		tag_fd = nltk.FreqDist(tag for (word, tag) in txt_tagged)
		tagged_freq = tag_fd.most_common()
		freq = {}
		for (tag, count) in tagged_freq:
			freq[tag] = np.sqrt(count / len(txt_tagged))
		return freq

	def get_sqrt_frequency_tags(self,df):
		'''
			Iterate through the articles, on each iteration, calls get_sqrt_frequency_tags. 
		'''
		df_pos = pd.DataFrame()
		
		for i, row in df.iteritems():
			dic_pos = {}
			for (key, val) in self.get_sqrt_frequency_tags_per_article(row).items():
				# if key+"_sqrtfreq" not in dic_pos.keys():
				# 	dic_pos[key+"_sqrtfreq"] = []
				dic_pos[key+"_sqrtfreq"] = val
			df_pos = df_pos.append(dic_pos, ignore_index=True)
		df_pos['id'] = df.index
		# df_pos =  df_pos.set_index('id')
		return df_pos
		

		
	def get_brunet_index(self, tokens):
		lm = self.get_MLELM(tokens)
		total_len = len(tokens)
		vocab_len = len(lm.vocab)
		return total_len / (vocab_len ** -0.165)

   
	def get_honore_statistic(self, tokens):
		lm = self.get_MLELM(tokens)
		vocab_len = len(lm.vocab)
		freq = Counter()
		for word in tokens:
			freq[word] += 1
		words_once = [word for (word, val) in freq.items() if val == 1]
		return np.log(len(tokens)/(1-len(words_once)/len(lm.vocab)))

	def count_pronouns(self, text):
		total = 0
		for pronoun in PRONOUNS:
			total += text.count(pronoun)
		return total

	def get_question_ratio(self, text):
		total = 0
		for q in QUESTION:
			total += text.count(q)

		if len(text) == 0:
			return 0
		else:
			return total/len(text)

	def get_personal_pronouns(self, df = None):
		if df is None:
			df = self.df
		df = self.reshapedf(df)
		pronoun_counts = pd.DataFrame()
		pronoun_counts['count'] = df['text'].apply(lambda x: self.count_pronouns(x))
		pronoun_counts['label'] = df['label']
		return pronoun_counts
		
	def get_word_count(self, tokens):
		length = len(tokens.split())
		if length == 0:
			print(tokens)
			return 0
		return sum(len(word) for word in tokens)/length

	def get_tfidf(self, df):
		X_train, X_test, y_train, y_test = train_test_split(df.drop(['label'], axis = 1), df.label, random_state=0)
		transformer = TfidfTransformer(smooth_idf = False)
		count_vectorizer = CountVectorizer(ngram_range=(1, 2))
		counts = count_vectorizer.fit_transform(X_train['text'].values)
		tfidf = transformer.fit_transform(counts)
		test_counts = count_vectorizer.transform(X_test['text'].values)
		test_tfidf = transformer.fit_transform(test_counts)
		return tfidf, test_tfidf, y_train, y_test


	#def getDistinctWords(tokens):
	#	tokenizer = RegexpTokenizer(r'\w+')
	#	zen_no_punc = tokenizer.tokenize(tokens)
	#	return len(set(w.title() for w in zen_no_punc if w.lower() not in stopwords.words()))

	#def get_average_word_length(self, tokens):
	#    Total characters / Total words
	#	article_length = len(tokens)
	#	average = sum(len(word) for word in tokens) / article_length
	#	return average

	#def get_article_length(self, tokens):
	#	article_length = len(tokens)
	#	return article_length

	#def getAvgSentenceLength(tokens,article):
	#	# In # of words
	#	tokenizer = RegexpTokenizer(r'\w+')
	#	zen_no_punc = tokenizer.tokenize(tokens)
	#	sentences = sentenceDictionary[article]
	#	return (float(len(zen_no_punc)/sentences))


	def get_syntactic_features(self, df = None):
		if df is None:
			df = self.df

		corpus = self.reshapedf(df)
		df_li = pd.DataFrame()

		df_li['brunetIndex'] = corpus['text'].apply(lambda x: self.get_brunet_index(x))
		df_li['honoreStatistic'] = corpus['text'].apply(lambda x: self.get_honore_statistic(x))
		df_li['questionRatio'] = corpus['text'].apply(lambda x: self.get_question_ratio(x))
		df_li['entropy'] = corpus['text'].apply(lambda x: self.get_standardized_word_entropy(x))		
		df_li['label'] = corpus['label']
		df_li['id'] = corpus.index

		df_pos = self.get_sqrt_frequency_tags(corpus['text'])
		df_pos = df_pos.fillna(0.)
		return pd.merge(df_li, df_pos, how = 'outer', on = 'id')

	def get_lexical_features(self, df = None):
		if df is None:
			df = self.df
		corpus = self.reshapedf(df)
		df_li = pd.DataFrame()

		df_li['articleLength'] = corpus['text'].str.split().apply(len)
		df_li['avgWordLength'] = corpus['text'].apply(lambda x: self.get_word_count(x))
		return df_li


	def get_all_features(self, df = None):
		if df is None:
			df = self.df

		print("Getting pyscholingustic features...")
		pronouns = self.get_personal_pronouns()
		print("Getting syntactic features...")
		syntactic = self.get_syntactic_features()
		print("Getting lexical features...")
		lexical = self.get_lexical_features()
		pronouns['id'] = pronouns.index
		syntactic['id'] = syntactic.index
		lexical['id'] = lexical.index
		label = pronouns.label
		pronouns = pronouns.drop(['label'], axis = 1)
		syntactic = syntactic.drop(['label'], axis = 1)

		feature_df = pd.merge(pronouns, syntactic, how = 'outer', on = 'id')
		feature_df = pd.merge(feature_df, lexical, how = 'outer', on = 'id')
		feature_df['label'] = label
		return feature_df
