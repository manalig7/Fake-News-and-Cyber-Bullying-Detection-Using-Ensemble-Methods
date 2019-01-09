
import pandas as pd  
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB



tsv = 'dataset/finaldataset_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append(ls[0].decode('utf-8'))
	y_train.append(int(ls[1]))
f.close()

m=len(x_train)
print (m)

data = x_train

tsv1 = 'dataset/finaldataset_test.txt'
f=open(tsv1,'r')

x_test=[]
y_test=[]

for line in f:
	ls=line.split('\t')
	x_test.append(ls[0].decode('utf-8'))
	data.append(ls[0].decode('utf-8'))
	y_test.append(int(ls[1]))
f.close()



print "TfIdf-Vectors"

tvec = TfidfVectorizer(decode_error='ignore')
features2 = tvec.fit_transform(data).toarray()

train_features = []
train_features = features2[:m]
#print(m)
#print(len(y_test))

test_features = []
test_features = features2[m:]
#print(test_features.shape)


ch2 = SelectKBest(chi2, k=10000)
train_chi2 = ch2.fit_transform(train_features, y_train)
test_chi2 = ch2.transform(test_features)

print(test_chi2.shape)

"""
This file contains the code for feature selecion for writeprints dataset as well as the application of the random forest classifier
"""
#Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

#Import the required packages

#Import pandas to read csv import pandas

#Import numpy for array related operations import numpy

#Import sklearn's feature selection algorithm

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

model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 50)
rfe = rfe.fit(X, Y)
# summarize the selection of the attributes


#Summarize scores 

np.set_printoptions(precision=3)
print("-------------")

print("summarized scores")

print(rfe.support_)

selected_feature_names=[]

for i in range(0,len(rfe.support_)):
	if rfe.support_[i]==True:
		selected_feature_names.append(i)

"""
print "Selected Features"
for i in range(0,len(selected_feature_names)):
	print selected_feature_names[i], selected_feature_scores[i]
"""
#We have the selected feature names now
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

### Combine both word vector and Stylometric features ###

"""print(train_chi2.shape)
print(test_chi2.shape)

print(len(feature_set[0]))
print(len(feature_set_test[0]))


#print(train_chi2[0].tolist()+feature_set[0])
#print(feature_set[0])
"""

final_train=[]
final_test=[]

for i in range(train_chi2.shape[0]):
	print(i)
	final_train.append(train_chi2[i].tolist()+feature_set[i])

for i in range(test_features.shape[0]):
	print(i)
	final_test.append(test_chi2[i].tolist()+feature_set_test[i])

#print(len(final_train[0]))
#print(len(final_test[0]))

### Voting method for classification ###

clf1 = MultinomialNB()
clf2 = LogisticRegression(random_state=0, solver='lbfgs')
clf3 = RandomForestClassifier()
clf4 = BaggingClassifier(GaussianNB(),max_samples=0.4, max_features=0.8,n_estimators=300)
clf5 = AdaBoostClassifier(n_estimators=300)

eclf1 = VotingClassifier(estimators=[('bag', clf4), ('lr', clf2), ('ada', clf5)], voting='hard')
# ('mnb', clf1), ('lr', clf2), ('rf', clf3) ,('ada', clf5), ('bag', clf4)
eclf1 = eclf1.fit(final_train, y_train)

print "\nAccuracy on Training Set :"
print eclf1.score(final_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print eclf1.score(final_test, y_test)

y_pred=eclf1.predict(final_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

