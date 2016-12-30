import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer #lemmatizer ensures 'nltk.stem' stems to full words 
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000000

'''
Create the Lexicon 'list' from a Language, derived from 'Pos' and 'Neg' files
'''
def create_lexicon(pos,neg):
	lexicon = []

	for fi in [pos,neg]: # collect all lexicon from file
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon] #reduce word lexicon to lemmatized
	w_counts = Counter(lexicon) # dict of word frequency  {'the:5252, 'and':434, ...}
	
	l2 = [] # l2 is final lexicon
	for w in w_counts:
		#print(w_counts[w])
		if 1000 > w_counts[w] > 50: # filter out word frequency to high or low, words like 'the, and, of' 
			l2.append(w)
	print('My language l2 is length: ', len(l2))
	return l2


def sample_handling(sample, lexicon, classification):

	featureset = []

	'''
		Return classified samples, as a list of the language with [1,0] as [pos, neg] classification

					    features      pos neg	
		featureset = [
						[1 0 0 1 1 0], [1 0],
						[0 1 0 1 0 0], [0 1],
						 ...
					  ]
	'''

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words] # stems to whole words
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features) # make unique
			featureset.append([features, classification])
	#print('featureset: ', featureset)
	return featureset

# Assign positive and negative to text files, 
def create_feature_sets_and_labels(pos,neg,test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1,0]) # positive is 1 0, negative is 0 1
	features += sample_handling('neg.txt', lexicon, [0,1])

	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))

	# Train on first 10% of testing_size, test on remainder 90%  
	train_x = list(features[:,0][:-testing_size])  # numpy trick [:,0] take the 'zero'th element' of multi list list, to take from the feature nodes list and nto the pos/neg list
	train_y = list(features[:,1][:-testing_size])  

	test_x = list(features[:,0][:-testing_size:])  
	test_y = list(features[:,1][:-testing_size:])  


	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y

if __name__=='__main__':
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)