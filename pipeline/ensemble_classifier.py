import numpy as np 
import pandas as pd 
from pipeline.feature_extraction import *
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import StandardScaler
from statistics import *

class EnsembleClassifier(object):

	def __init__(self,
				 dense_features = None,
				 sparse_data = None,
				 meta_classifier = None,
				 meta_params = None):

		self.dense_features = dense_features
		self.sparse_data = sparse_data
		self.meta_classifier = meta_classifier
		self.meta_params = meta_params
		self.dense_data = None
		self.train_label = None
		self.test_label = None
		self.clfs = []
		self.data_type = []
		self.sub_data = pd.DataFrame()
		self.kf = KFold(n_splits = 5, random_state = 22, shuffle = False)

	def reset(self):
		self.sub_data = pd.DataFrame()


	def load_data(self, dense_features = None, sparse_data = None, CrossVal = False):
		if dense_features is not None:
			self.dense_features = dense_features
		if sparse_data is not None:
			self.sparse_data = sparse_data
		if CrossVal:
			X_train = self.dense_features[0].drop(['label'], axis = 1)
			X_test = self.dense_features[1].drop(['label'], axis = 1)
			y_train = self.dense_features[0].label
			y_test = self.dense_features[1].label
		else:
			X = self.dense_features.drop(['label'], axis = 1)
			y = self.dense_features.label
			X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False)
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)
		self.dense_data	= [X_train_scaled, X_test_scaled]
		self.train_label = y_train
		self.test_label	= y_test

	def add(self, classifier, classifier_type = None, data_type = None, params = None):
		if classifier_type == 'meta':
			self.meta_classifier = classifier
		else:
			self.clfs.append(classifier)
		if self.data_type is not None:
			self.data_type.append(data_type)


	def train_learners(self, proba):

		for i, clf in enumerate(self.clfs):
			y = self.train_label
			if self.data_type[i] == 'dense':
				X = self.dense_data[0]
			else:
				X = self.sparse_data[0]
			print(X)
			clf.fit(X, y)

			pred_list_a = []
			pred_list_b = []
			for train_index, test_index in self.kf.split(X):
				X_train_fold, X_test_fold = X[train_index], X[test_index]
				y_train_fold, y_test_fold = y[train_index], y[test_index]
				clf.fit(X_train_fold, y_train_fold)
				if proba:
					pred_list_a.extend([row[0] for row in clf.predict_proba(X_test_fold)])
					pred_list_b.extend([row[1] for row in clf.predict_proba(X_test_fold)])
				else:
					pred_list_a.extend(clf.predict(X_test_fold))
			if proba:
				self.sub_data['predictions_a_' + str(i)] = pred_list_a
				self.sub_data['predictions_b_' + str(i)] = pred_list_b
			else:
				self.sub_data['predictions_' + str(i)] = pred_list_a
		return self.sub_data


	def train_meta(self, proba):
		self.meta_classifier.fit(self.sub_data, self.train_label)
		return self.meta_classifier

	def fit(self, proba = True):
		self.load_data()
		self.train_learners(proba)
		self.train_meta(proba)


	def predict(self, proba = True):
		self.sub_data = pd.DataFrame()
		y = self.test_label

		for i, clf in enumerate(self.clfs):
			if self.data_type[i] == 'dense':
				X = self.dense_data[1]
			else:
				X = self.sparse_data[1]

			if proba:
				self.sub_data['predictions_a' + str(i)] = [row[0] for row in clf.predict_proba(X)]
				self.sub_data['predictions_b' + str(i)] = [row[1] for row in clf.predict_proba(X)]
			else:
				self.sub_data['predictions_' + str(i)] = clf.predict(X)

		return self.meta_classifier.score(self.sub_data, y)

	def run_cross_val(self, file = 'data/processed/datasets/data_preprocessed_final.csv', proba = True):
		df = pd.read_csv(file)[:10]
		df = df.dropna()
		#df = df.reset_index(["Unnamed: 0", "id"]).drop(["Unnamed: 0"], axis = 1)
		k_fold = KFold(n_splits = 3, random_state = 22)
		scores = []
		for train_index, test_index in k_fold.split(df):
			self.reset()
			kg = TextFeatures(df)
			train_features = kg.get_all_features(df.iloc[train_index])
			test_features = kg.get_all_features(df.iloc[test_index])
			text = kg.get_tfidf(df)
			train_text = text[train_index]
			test_text = text[test_index]
			self.load_data(dense_features = [train_features, test_features],
						   sparse_data = [train_text, test_text], CrossVal = True)
			self.train_learners(proba)
			self.train_meta(proba)
			scores.append(self.predict(proba))
		return mean(scores)

		#if feature_file is None:
		#	kg = TestFeatures(df)
		#	features = kg.get_all_features()