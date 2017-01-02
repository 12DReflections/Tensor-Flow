

import os 
import pandas 

'''
The variables that are to be decided by the user are 

Inputs (<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>)
Outputs (<CLOSE>)


Percentage of training data (60%)
Percentage of testing data (40%)

Date range: Jan 2001 - 30 Nov 2016
'''

def main():

	# train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
	# with open('sentiment_set.pickle', 'wb') as f:
	# 	pickle.dump([train_x, train_y, test_x, test_y], f)

	# dir_path = os.path.dirname(os.path.realpath(__file__))
	# print(dir_path)
	

	time_curr, open_curr, high_curr, low_curr, close_curr =	data_input()
	print(time_curr)
	


	# create_feature_sets_and_labels('pos.txt', 'neg.txt')
	# create_feature_sets_and_labels()
	

# Read in data using PANDAS from csv
def data_input():

	'''
	Data Example
	<TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
	AUDJPY,20010102,230100,64.30,64.30,64.30,64.30,4
	'''

	colnames = ['TICKER', 'DTYYYYMMDD', 'TIME','OPEN','HIGH','LOW','CLOSE', 'VOL']

#	colnames = ['TICKER', 'DATE', 'TIME','OPEN','HIGH','LOW','CLOSE']
	data = pandas.read_csv('data/test.txt', names=colnames)

	time_curr = data.TIME.tolist()
	open_curr = data.OPEN.tolist()
	high_curr = data.HIGH.tolist()
	low_curr = data.LOW.tolist()
	close_curr = data.CLOSE.tolist()
	return time_curr, open_curr, high_curr, low_curr, close_curr














def sample_handling(sample, lexicon, classification):

	featureset = []

	'''
		Return classified samples, as a list of the language with [1,0] as [end, neg] classification

						features	  pos neg	
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

# Assign "pos/neg" to training Train and Test
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
	main()