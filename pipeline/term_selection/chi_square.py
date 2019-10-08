from tfidf import file_dict, evc_dict, tfidf, tfs, token_dict, feature_names
from nltk.stem.porter import PorterStemmer
import nltk

terms = []
for key in token_dict:
	tokens = nltk.word_tokenize(token_dict[key])
	for term in tokens:
		terms.append(term)

score = {}
stemmed_terms = []
doc_count = 0
'''
for term in terms:
	stem = PorterStemmer().stem(term)
	if stem not in stemmed_terms:
		stemmed_terms.append(stem)
		A = 0
		B = 0
		C = 0
		D = 0
		for key in file_dict:
			try:
				if evc_dict[key[:-4]] == 1:
					if stem in file_dict[key]:
						A += 1
					else:
						C += 1
				else:
					if stem in file_dict[key]:
						B += 1
					else:
						D += 1
				doc_count += 1
			except KeyError:
				continue

		chi_square = (doc_count * (A * D - C * B) ** 2) / ((A + C) * (B + D) + (A + B) * (C + D))
		score[term] = chi_square


value = 'zte'
for item in score:
	if score[item] > score[value]:
		value = item
print(value)
'''
#COMPLETELY STEMMED

score = {}
for term in feature_names:
	A = 0
	B = 0
	C = 0
	D = 0
	for key in file_dict:
		try:
			if evc_dict[key[:-4]] == 1:
				if term in file_dict[key]:
					A += 1
				else:
					C += 1
			else:
				if term in file_dict[key]:
					B += 1
				else:
					D += 1
			doc_count += 1
		except KeyError:
			continue

	chi_square = (doc_count * (A * D - C * B) ** 2) / ((A + C) * (B + D) + (A + B) * (C + D))
	score[term] = chi_square
#print(score)
value = 'yike'
for item in score:
	if score[item] > score[value]:
		value = item
#print(value)


sorted_score = {}
for key, value in sorted(score.items(), key=lambda x: x[1], reverse = True):
    sorted_score[key] = value
#print(sorted_score)