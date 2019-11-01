from pipeline.feature_extraction import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from sklearn.preprocessing import label_binarize


df = pd.read_csv("data/processed/datasets/data_preprocessed.csv")
df = df.iloc[:10000]
df = df.dropna()
kg = TextFeatures(df)
df_reshaped = kg.reshapedf()
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 22)
#cvec = CountVectorizer()
#tvec = TfidfVectorizer()
mdl = LogisticRegression(random_state = 22)

acc_df1 = pd.DataFrame()

def count_vec_ngram(params, X_train, y_train, mdl):
	cvec_p = CountVectorizer(ngram_range = (params))
	cvec_p.fit(X_train.values)
	X_train_cvec_p = cvec_p.transform(X_train.values)
	transformer = TfidfTransformer(smooth_idf = False)
	tfidf = transformer.fit_transform(X_train_cvec_p)
	cvec_score_p = cross_val_score(mdl, tfidf, y_train, cv = 3)
	print(params)
	return cvec_score_p.mean()

params = [(1,1),(1,2), (1,3)]
ngram_scores = []
for p in params:
	ngram_scores.append(count_vec_ngram(p, X_train, y_train, mdl))

ngrams = ['cvec gram_1', 'cvec_gram 2', 'cvec_gram 3']
ngram_df = pd.DataFrame({'params': ngrams, 'scores':ngram_scores}, index = [0,1,2])
acc_df1['params'] = ngram_df['params']
acc_df1['scores'] = ngram_df['scores']
print(acc_df1)

sns.pointplot(x = 'params', y = 'scores', data = ngram_df)
plt.ylabel('Accuracy Score')
plt.xlabel('ngrams')
plt.xticks(rotation = 40)
plt.show()

def count_vec_max_features(params, X_train, y_train, mdl):
	cvec_p = CountVectorizer(max_features=params) 

	cvec_p.fit(X_train.values)
	X_train_cvec_p = cvec_p.transform(X_train.values)
	transformer = TfidfTransformer(smooth_idf = False)
	tfidf = transformer.fit_transform(X_train_cvec_p)
	# cross val score/ predict
	cvec_score_p = cross_val_score(mdl, tfidf, y_train, cv=3)

	# cross validation 
	return cvec_score_p.mean()

mf_params = [None, 500, 1000, 5000, 10000]
max_features_scores = [count_vec_max_features(p, X_train, y_train, mdl) for p in mf_params]
max_features = ['max_f_'+str(p) for p in mf_params]

# dataframe for scores
max_features_df = pd.DataFrame({'params':max_features, 'scores':max_features_scores})

sns.pointplot(x='params', y='scores', data =max_features_df)
plt.ylabel('Accuracy Score')
plt.xlabel('Max Features')
plt.xticks(rotation=40)
plt.title('Accuracy of Max Features')
plt.show()

acc_df2 = acc_df1.append(max_features_df.drop(max_features_df.index[[1,2]]))
acc_df2.reset_index(inplace=True, drop=True)
print(acc_df2)

def count_vec_max_df(params, X_train, y_train, mdl):
	cvec_p = CountVectorizer(max_df=params) 

	cvec_p.fit(X_train.values)
	X_train_cvec_p = cvec_p.transform(X_train.values)
	transformer = TfidfTransformer(smooth_idf = False)
	tfidf = transformer.fit_transform(X_train_cvec_p)
	# cross val score/ predict
	cvec_score_p = cross_val_score(mdl, tfidf, y_train, cv=3)

	# cross validation 
	return cvec_score_p.mean()

mdf_params = [0.25, 0.5, 0.75, 1.0]
max_df_scores = [count_vec_max_df(p, X_train, y_train, mdl) for p in mdf_params]
max_df = ['max_df_'+str(p) for p in mdf_params]

# dataframe for scores
max_df_df = pd.DataFrame({'params':max_df, 'scores':max_df_scores}, index=[0,1,2,3])
acc_df3 = acc_df2.append(max_df_df.iloc[:2,:])
acc_df3.reset_index(inplace=True, drop=True)
print(acc_df3)

sns.pointplot(x='params', y='scores', data =acc_df3)
plt.ylabel('Accuracy Score')
plt.xlabel('max_df')
plt.xticks(rotation=40)
plt.title('Accuracy of Max df')
plt.show()

cvec_p = CountVectorizer(ngram_range=(1,3)) 
cvec_p.fit(X_train.values)
X_train_cvec_p = cvec_p.transform(X_train.values)
transformer = TfidfTransformer(smooth_idf = False)
tfidf = transformer.fit_transform(X_train_cvec_p)
mdl.fit(tfidf, y_train)

def print_top10(vectorizer, clf, class_labels):
	"""Prints features with the highest coefficient values, per class"""
	feature_names = vectorizer.get_feature_names()
	
	for i, class_label in enumerate(class_labels):
#         output the original index of the top 10 coef
		if class_label == 1:
			top10 = np.argsort(clf.coef_[0])[-10:]
		else:
			top10 = np.argsort(clf.coef_[0])[:10]

		print("%s: %s" % (class_label,
			  ", ".join(feature_names[j] for j in top10)))


feature_df = pd.DataFrame()
feature_names = cvec_p.get_feature_names()
for i, class_label in enumerate([0,1]):
	if class_label == 1:
		index = np.argsort(mdl.coef_[0])
	else:
		index = np.argsort(-mdl.coef_[0])
	feature_df['feature_names'] = [feature_names[j] for j in index]
	if class_label == 1:
		class_ = mdl.coef_[0]
	else:
		class_ = -mdl.coef_[0]
	feature_df['class'+str(class_label)] = np.abs(class_)
		
print(feature_df.head())


class_0_abs_10 = feature_df.sort_values('class0', ascending=False).head(10)
class_0_abs_10.plot(x="feature_names", y=["class0", "class1"], kind="bar")
plt.ylabel('absolute coefficients')
plt.title('Top 10 Real Class Features Comparing to Other Classes')
plt.show()
print_top10(cvec_p, mdl, [0,1])


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')



def plot_precision_recall(model,y_bin,X,f1_lines=True):
	precision, recall, _ = metrics.precision_recall_curve(y_bin[:,0],model.predict_proba(X)[:,0])
	plt.figure(figsize=(6,4))
	plt.plot(precision,recall,lw = 2)

	plt.ylim([0,1.05])
	plt.legend(loc='lower left')
	plt.title('Precision-Recall Curve',fontsize=20)
	plt.xlabel('Recall',fontsize=18)
	plt.ylabel('Precision',fontsize=18)
	
	if f1_lines == True:
		for const in [0.2,0.4,0.6,0.8]:
			x_vals = np.linspace(0.001,0.999,100)
			y_vals = 1./(2./const-1./x_vals)
			plt.plot(x_vals[y_vals>0],y_vals[y_vals>0],color='lightblue',ls='--',alpha=0.9)
			plt.ylim([0,1])
			plt.annotate('f1={0:0.1f}'.format(const), xy=(x_vals[-10], y_vals[-2]+0.0))

	plt.show()

def plot_roc(model,y_bin,X):
	fpr, tpr, _ = metrics.roc_curve(y_bin[:,0],model.predict_proba(X)[:,0])
	plt.plot(fpr,tpr,lw=2)
	plt.plot([0,1],[0,1],ls='--',lw=2)
	plt.ylim([0,1.05])
	plt.legend(loc='lower right')
	plt.title('ROC Curve',fontsize=20)
	plt.xlabel('FPR',fontsize=18)
	plt.ylabel('TPR',fontsize=18)
	
	plt.show()


predictions_estimator = cross_val_predict(mdl,tfidf,y_train,cv=5)

# Compute confusion matrix
cm_cv = confusion_matrix(y_train,predictions_estimator)
x_class = ['True', 'Fake']

# Plot non-normalized confusion matrix
plot_confusion_matrix(cm_cv, classes=x_class, title='Confusion matrix, without normalization')
plt.show()

# Plot normalized confusion matrix
plot_confusion_matrix(cm_cv, classes=x_class, normalize=True, title='Normalized confusion matrix')
plt.show()

print(classification_report(y_train,predictions_estimator))
y_bin = label_binarize(y_train,mdl.classes_)
plot_precision_recall(mdl,y_bin,tfidf)
plot_roc(mdl,y_bin,tfidf)