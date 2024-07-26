from utils import *
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def naive_bayes():
	percentage_positive_instances_train = 0.2
	percentage_negative_instances_train = 0.2

	percentage_positive_instances_test  = 0.2
	percentage_negative_instances_test  = 0.2
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)


	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))

	pos_train_voc_size = sum(len(train_set) for train_set in pos_train)
	neg_train_voc_size = sum(len(train_set) for train_set in neg_train)

	word_freq_prob = defaultdict()
	word_pos_freq = defaultdict(int)
	word_neg_freq = defaultdict(int)

	for train_sets in pos_train:
		for word in train_sets:
			word_pos_freq[word] += 1
	for train_sets in neg_train:
		for word in train_sets:
			word_neg_freq[word] += 1
	
	
	for word in vocab:
		word_freq_prob[word] = [word_pos_freq[word], word_pos_freq[word] / pos_train_voc_size, word_neg_freq[word], word_neg_freq[word] / neg_train_voc_size]
	
	alpha = 0.0001
	
	accuracy = []
	x = []
	while(alpha <= 10000):
		confusion_matrix = get_confusion_matrix(word_freq_prob, pos_test, neg_test, len(pos_train), len(neg_train), pos_train_voc_size, neg_train_voc_size, len(vocab), alpha)
		total_count = confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[1, 0]
		accuracy.append(((confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 1]) * 100) / total_count)
		x.append(alpha)
		alpha *= 10
		
    
	fig, ax = plt.subplots()
    # Plotting the line graph
	ax.plot(np.log10(x), accuracy, label='accuracy')

    # Adding title
	ax.set_title('Graph of accuracy vs log(alpha)')

    # Adding labels to the axes
	ax.set_xlabel('log(alpha)')
	ax.set_ylabel('accuracy in %')
	
	ax.grid(True)

    # Display the plot
	plt.show()
		


	with open('vocab.txt','w') as f:
		for word in vocab:
			f.write("%s\n" % word)
	

def get_confusion_matrix(word_freq_prob, pos_test, neg_test, pos_train_size, neg_train_size, pos_voc_size, neg_voc_size, vocab_size, alpha):
	"""
	returns the confusion matrix based on the trained model of pos and neg train classes when testing on pos_test and neg_test dataset
	word_freq_prob -> key value store used for storing word -> [freq in pos_train dataset, 
                                                                probability of occurance in pos_train_dataset, 
                                                                freq in neg_train dataset, 
                                                                probability of occurance in pos_train_dataset] 
	pos_test -> positive class test dataset
	neg_test -> neg class test dataset
	pos_train_size -> size of the positive class training dataset
	neg_train_size -> size of the negative class training dataset
	pos_voc_size -> size of the vocabulary in pos training dataset
	neg_voc_size -> size of the vocabulary in neg training dataset
	voc_size -> number of unique words in training dataset
	alpha -> missing words weight parameter
	"""
	correct_count = 0
	wrong_count = 0
	confusion_matrix = pd.DataFrame(np.nan, index=[0, 1], columns=['pred_pos', 'pred_neg'])

	for test_set in pos_test:
		pos_prob = 0
		neg_prob = 0
		pos_prob = get_prob(word_freq_prob, test_set, pos_train_size, neg_train_size, pos_voc_size, neg_voc_size, vocab_size, alpha, True)
		neg_prob = get_prob(word_freq_prob, test_set, pos_train_size, neg_train_size, pos_voc_size, neg_voc_size, vocab_size, alpha, False)

		if pos_prob > neg_prob:
			correct_count += 1
		elif pos_prob < neg_prob:
			wrong_count += 1
		else:
			if random.choice([0, 1]) == 0:
				correct_count += 1
			else:
				wrong_count += 1

    #True label is positive and predicited is also positive
	confusion_matrix.loc[0, 'pred_pos'] = correct_count
	#True label is positive but predicated is negative
	confusion_matrix.loc[0, 'pred_neg'] = wrong_count

	correct_count = 0
	wrong_count = 0
	
	for test_set in neg_test:
		pos_prob = get_prob(word_freq_prob, test_set, pos_train_size, neg_train_size, pos_voc_size, neg_voc_size, vocab_size, alpha, True)
		neg_prob = get_prob(word_freq_prob, test_set, pos_train_size, neg_train_size, pos_voc_size, neg_voc_size, vocab_size, alpha, False)

		if pos_prob < neg_prob:
			correct_count += 1
		elif pos_prob > neg_prob:
			wrong_count += 1
		else:
			if random.choice([0, 1]) == 0:
				correct_count += 1
			else:
				wrong_count += 1

    #True label is negative and predicited is also negative
	confusion_matrix.loc[1, 'pred_neg'] = correct_count
	#True label is negative but predicited is positive
	confusion_matrix.loc[1, 'pred_pos'] = wrong_count

	return confusion_matrix

def get_prob(word_freq_prob, test, pos_size, neg_size, pos_voc_size, neg_voc_size, vocab_size, alpha, is_pos):
	unique_words = set(test)
	
	prob = 0
	if is_pos:
		# probability that it is a positive class
		prob += math.log(pos_size / (pos_size + neg_size))
		missing_count = 0
		for word in unique_words:
			if word not in word_freq_prob or word_freq_prob[word][0] == 0:
				missing_count +=1
		for word in unique_words:
			if word in word_freq_prob:
				prob += math.log((word_freq_prob[word][0] + alpha) / (pos_voc_size + (alpha * (vocab_size + missing_count))))
			else:
				prob += math.log(alpha / (pos_voc_size + (alpha * (vocab_size + missing_count))))
	else:
		prob += math.log(neg_size / (pos_size + neg_size))
		missing_count = 0
		for word in unique_words:
			if word not in word_freq_prob or word_freq_prob[word][2] == 0:
				missing_count +=1
		for word in unique_words:
			if word in word_freq_prob:
				prob += math.log((word_freq_prob[word][2] + alpha) / (neg_voc_size + (alpha * (vocab_size + missing_count))))
			else:
				prob += math.log(alpha / (neg_voc_size + (alpha * (vocab_size + missing_count))))
	return prob

if __name__=="__main__":
	naive_bayes()
