#Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

#Import the required packages

#Import pandas to read csv import pandas

#Import numpy for array related operations import numpy

#Import sklearn's feature selection algorithm

from sklearn.feature_selection import SelectKBest
import pandas as pd  
import numpy as np
#Import chi2 for performing chi square test 
from sklearn.feature_selection import chi2



#URL for loading the dataset


#Define the attribute names

#names = ['charcount', 'digratio', 'upratio', 'whiteratio', 'freq_letters', 'freq_sp_char', 'wordcount', 'shortratio', 'word_char_ratio','avg_sentence_length', 'avg_sen_len_words', 'total_diff_words', 'freq_once_ocurring_words', 'freq_twice_ocurring_words', 'word_length_freq_dist', 'num_punc', 'func_count', 'num_lines','num_sentences','num_paras','num_sen_per_para','num_char_per_para','num_word_per_para','greeting','num_quotes','has_url','freq_content_Sp_words','yule_measure','fake']
names=[]
for i in range(0,74):
	names.append(i)
print("names done")
#Create pandas data frame by loading the data from URL

dataframe = pd.read_csv("writeprints_features_final.tsv",sep='\t', names=names)
print("file read")
#Create array from data values

array = dataframe.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
X = array[:,0:73]

Y = array[:,73]
print("attributes Split")

#We will select the features using chi square

test = SelectKBest(score_func=chi2, k=30)
print("slected k best")

#Fit the function for ranking the features by score

fit = test.fit(X, Y)

#Summarize scores 

np.set_printoptions(precision=3)
print("-------------")

print("summarized scores")

print(fit.scores_)

fit.scores_=np.ndarray.tolist(fit.scores_)

print "Sorted Scores"
print sorted(fit.scores_)

selected_feature_names=[]
selected_feature_scores=[]

while len(selected_feature_names)<30:
	maxi=-1
	index=-1
	for i in range(0,len(fit.scores_)):
		if fit.scores_[i]>maxi:
			maxi=fit.scores_[i]
			index=i
	selected_feature_names.append(names[index])
	selected_feature_scores.append(fit.scores_[index])
	names.pop(index)
	fit.scores_.pop(index)

print "Selected Features"
for i in range(0,len(selected_feature_names)):
	print selected_feature_names[i], selected_feature_scores[i]


"""
print("summarized scores")
#Apply the transformation on to dataset
"""
"""
features = fit.transform(X)
print("applied transformation")
"""
#Summarize selected features 
"""
print("-------------")
print("SELECTED FEATURES:")
print(features[0:5,:])
"""