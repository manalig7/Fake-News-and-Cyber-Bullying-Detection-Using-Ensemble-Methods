#Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

#Import the required packages

#Import pandas to read csv import pandas

#Import numpy for array related operations import numpy

#Import sklearn's feature selection algorithm

from sklearn.feature_selection import SelectKBest
import pandas as pd  
import numpy as np
import sys
#Import chi2 for performing chi square test 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#URL for loading the dataset


#Define the attribute names

#names = ['charcount', 'digratio', 'upratio', 'whiteratio', 'freq_letters', 'freq_sp_char', 'wordcount', 'shortratio', 'word_char_ratio','avg_sentence_length', 'avg_sen_len_words', 'total_diff_words', 'freq_once_ocurring_words', 'freq_twice_ocurring_words', 'word_length_freq_dist', 'num_punc', 'func_count', 'num_lines','num_sentences','num_paras','num_sen_per_para','num_char_per_para','num_word_per_para','greeting','num_quotes','has_url','freq_content_Sp_words','yule_measure','fake']
names=[]
for i in range(0,74):
	names.append(i)
#Create pandas data frame by loading the data from URL

dataframe = pd.read_csv("writeprints_features_final.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
X = array[:,0:73]

Y = array[:,73]

model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 50)
rfe = rfe.fit(X, Y)
# summarize the selection of the attributes

np.set_printoptions(precision=3)


selected_feature_names=[]

for i in range(0,len(rfe.support_)):
	if rfe.support_[i]==True:
		selected_feature_names.append(i)


feature_set_temp=X

feature_set_temp=np.ndarray.tolist(feature_set_temp)

y_train=Y
feature_set=[]
for i in range(0,len(X)):
	feature_set.append([])

#We need to remove the undesired features from the feature set
for item in selected_feature_names:
	for j in range(0,len(feature_set)):
		feature_set[j].append(feature_set_temp[j][item])

names=[]
for i in range(0,74):
	names.append(i)

dataframe1 = pd.read_csv("writeprints_features_final_test.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe1.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset

X_test = array[:,0:73]

Y_test = array[:,73]

feature_set_test_temp=X_test

feature_set_test_temp=np.ndarray.tolist(feature_set_test_temp)

y_test=Y_test
feature_set_test=[]
for i in range(0,len(X_test)):
	feature_set_test.append([])

#We need to remove the undesired features from the feature set
for item in selected_feature_names:
	for j in range(0,len(feature_set_test)):
		feature_set_test[j].append(feature_set_test_temp[j][item])

n_writeprints=10
size_writeprints=700

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import random
clf = RandomForestClassifier()
predictions=[]
for p in range(0,len(feature_set_test)):
	predictions.append([])

#Need to Select different subsets
for i in range(0,n_writeprints):
	temp_list=[]
	temp_list_y=[]
	for j in range(0,size_writeprints):
		num=random.randint(0,len(feature_set)-1)
		temp_list.append(feature_set[num])
		temp_list_y.append(y_train[num])
	clf.fit(temp_list,temp_list_y)
	pred=clf.predict(feature_set_test)
	for p in range(0,len(pred)):
		predictions[p].append(pred[p])

for i in range(0,len(predictions)):
	for j in range(0,len(predictions[i])):
		predictions[i][j]=int(predictions[i][j])

final_pred=[]

for i in range(0,len(predictions)):
	#print predictions[i]
	final_pred.append(max(set(predictions[i]), key=predictions[i].count))

print "Accuracy on Testing Set"
acc=0
total=0
for i in range(0,len(y_test)):
	if y_test[i]==final_pred[i]:
		acc=acc+1
	total=total+1

accuracy=float(acc)/float(total)
print accuracy
print ("\nPrecision Score")
print (precision_score(y_test, final_pred))
print ("\nRecall Score")
print (recall_score(y_test, final_pred))
print ("\nF1 Score")
print (f1_score(y_test, final_pred))

