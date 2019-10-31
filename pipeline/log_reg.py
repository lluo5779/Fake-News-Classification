from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
class FeaturePredictor(object):

	def __init__(self,
				 file = None,
				 features = ['count', 'brunetIndex', 'honoreStatistic'],
				 scaler = None):
		self.features = features
		self.scaler = scaler
		self.file = file

	def data_loader(self):
		df = pd.read_csv(self.file)
		df = df.drop(['id', 'fixme'], axis = 1)
		X = df[self.features]
		y = df['label']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 22)
		if self.scaler is not None:
			self.scaler.fit(X_train)
			X_train_scaled = self.scaler.transform(X_train)
			X_test_scaled = self.scaler.transform(X_test)
			return X_train_scaled, X_test_scaled, y_train, y_test
		return X_train, X_test, y_train, y_test

	
	def model(self):
		X_train, X_test, y_train, y_test = self.data_loader()
		mdl = LogisticRegression(random_state = 22)
		#mdl = RandomForestClassifier(random_state = 22)
		mdl.fit(X_train, y_train)
		#y_pred = mdl.predict(X_test)
		#return cm(y_test, y_pred)
		return mdl.score(X_test, y_test)
