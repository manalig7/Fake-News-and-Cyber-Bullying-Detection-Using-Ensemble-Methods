
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
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score


def fit_transform(d):
	res=[]
	for i in range(0,len(d)):
		temp=[]	
		for j in range(0,len(voc)):
			#print(voc[j])
			if voc[j] in d[i]:
				#print 	(np.mean(model_W2V.wv[voc[j]]))	
				temp.append(np.mean(model_W2V.wv[voc[j]]))
			else :
				temp.append(0)		
		res.append(temp)
	return res

x = []
##### training dataset #####

tsv = 'dataset/finaldataset_train.txt'
f=open(tsv,'r')
y_train=[]
data=[]
#lent=[]

tokenizer = RegexpTokenizer(' ', gaps=True)

for line in f :
	ls=line.split('\t')
	x.append(ls[0])
	temp = [] 
	#print(ls[0])
	for j in tokenizer.tokenize(ls[0].decode('utf-8')):
		#print(j) 
	       	temp.append(j) 
	data.append(temp)
	#lent.append(len(temp)) 
	y_train.append(int(ls[1]))
f.close()

m=len(x)

#print(max(lent))

##### testing dataset #####
tsv1 = 'dataset/finaldataset_test.txt'
f=open(tsv1,'r')
y_test=[]

#data1=[]
for line in f:
	ls=line.split('\t')
	x.append(ls[0])
	temp = [] 
	#print(ls[0])
	for j in tokenizer.tokenize(ls[0].decode('utf-8')):
		#print(j) 
	       	temp.append(j) 
	data.append(temp) 
	y_test.append(int(ls[1]))
f.close()

#pad_len=max(lent)

model_W2V = Word2Vec(data, size=10, window=5, min_count=1, workers=5, sg=1, max_vocab_size=10000)

print "CBOW W2V model_done!"

voc=list(model_W2V.wv.vocab)
print(len(voc))

XVAL=fit_transform(data)

print ("Transformed!!")

x_train=[]
x_train=XVAL[:m]
print(np.array(x_train).shape)
x_test = []
x_test=XVAL[m:]
print(np.array(x_test).shape)


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

#"""print(x_train.shape)
#print(x_test.shape)

print(len(feature_set[0]))
print(len(feature_set_test[0]))


#print(x_train[0]+feature_set[0])
#print(feature_set[0])


final_train=[]
final_test=[]

for i in range(len(x_train)):
	print(i)
	final_train.append(x_train[i]+feature_set[i])

for i in range(len(x_test)):
	print(i)
	final_test.append(x_test[i]+feature_set_test[i])

#print(len(final_train[0]))
#print(len(final_test[0]))"""

### Voting method for classification ###

clf1 = GaussianNB()
clf2 = LogisticRegression(random_state=0, solver='lbfgs')
clf3 = RandomForestClassifier()
clf4 = BaggingClassifier(GaussianNB(),max_samples=0.4, max_features=0.8,n_estimators=300)
clf5 = AdaBoostClassifier(n_estimators=300)

eclf1 = VotingClassifier(estimators=[('bag', clf4), ('lr', clf2), ('rf', clf3)], voting='hard')
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

