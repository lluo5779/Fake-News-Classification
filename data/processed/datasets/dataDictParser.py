import nltk, re
from textblob import Word

session_notes = open('cdlSessionNotes/raw_session_notes.txt', 'r', errors = 'ignore')
data_dictionary = open('dataDictionary/data_dictionary.txt', 'r', errors = 'ignore')
unrecognized_terms = open('dataDictionary/unrecognized_terms.txt', 'w')

unrecognized_term_set = set()
dictionary = {}
pattern = re.compile(r"(?:[$]?\w+(?:[-'?]\w+)*[%]?)") #add |(?:[,.?!/]) to capture punctuation, (?:\{.\})| for idk

for line in data_dictionary:
	line = line.replace("\n", "")
	if (line != ""):
		line = line.split(':')
		dictionary[line[0].lower()] = line[1]
		
data_dictionary.close()

def checkAdditional(token):
	if (len(Word(token).synsets) == 0 and not token.lower() in dictionary):
		return True

for line in session_notes:
	line = line.split('	')[6]
	tokens = pattern.findall(line)

	for token in tokens:
		token = re.sub('\'s$', '', token)
		if (not token.isdigit()):
			if (len(Word(token).synsets) == 0 and not token.lower() in dictionary):
				unrecognized_term_set.add(token.lower())

print(len(unrecognized_term_set))

for item in unrecognized_term_set:
	unrecognized_terms.write(item + '\n')
	
session_notes.close()
unrecognized_terms.close()
