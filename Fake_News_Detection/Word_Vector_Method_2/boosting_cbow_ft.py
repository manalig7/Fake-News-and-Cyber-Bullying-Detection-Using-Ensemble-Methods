#Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

#Import the required packages

#Import pandas to read csv import pandas

#Import numpy for array related operations import numpy

#Import sklearn's feature selection algorithm
from random import seed
import numpy as np
seed(20)
np.random.seed(20)
from sklearn.feature_selection import SelectKBest
import pandas as pd  
import gensim

import sys
#Import chi2 for performing chi square test 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
##########################FOR SQ W2V###########################
#############################################################
import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score


def fit_transform(d):
	res=[]
	for i in range(0,len(d)):
		temp=[]	
		for j in range(0,len(voc)):
			#print(voc[j])
			if voc[j] in d[i]:
				#print 	(np.mean(model_W2V.wv[voc[j]]))	
				temp.append(np.mean(model_FT.wv[voc[j]]))
			else :
				temp.append(0)		
		res.append(temp)
	return res


x = []
##### training dataset #####

tsv = 'finaldataset_train.txt'
f=open(tsv,'r')
y_train=[]
data=[]
lent=[]

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
	lent.append(len(temp)) 
	y_train.append(int(ls[1]))
f.close()

m=len(x)

#print(max(lent))

##### testing dataset #####
tsv1 = 'finaldataset_test.txt'
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
	lent.append(len(temp))  
	y_test.append(int(ls[1]))
f.close()

pad_len=max(lent)

model_FT = gensim.models.FastText.load("cbow_ft.model")
#model_W2V = Word2Vec(data, size=10, window=5, min_count=1, workers=5, sg=0,max_vocab_size=10000)

#print "SG W2V model_done!"

voc=list(model_FT.wv.vocab)
#	print(len(voc))
XVAL=fit_transform(data)

#print ("Transformed!!")

x_train=[]
x_train=XVAL[:m]
#print(np.array(x_train).shape)
x_test = []
x_test=XVAL[m:]
#print(np.array(x_test).shape)
x_train_sg=x_train
x_test_sg=x_test

##################################################################
##################################################################


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

#np.random.seed(20)
#model = LogisticRegression(np.random.seed(20))
#model=pickle.load(open('model_feature_selection', 'rb'))
#pickle.dump(model, open('model_feature_selection', 'wb'))
model = LogisticRegression(random_state=20)
# create the RFE model and select 3 attributes
rfe = RFE(model, 50)
rfe = rfe.fit(X, Y)

rankings= list(rfe.ranking_)
#print rankings.count(1)
#print rfe.support_
# summarize the selection of the attributes

np.set_printoptions(precision=3)


selected_feature_names=[]



for i in range(0,len(rfe.support_)):
	if rfe.support_[i]==True:
		selected_feature_names.append(i)

"""
p=open('selected_features','w')

for item in selected_feature_names:
	p.write(str(item)+'\n')

print selected_feature_names
"""
"""
p=open('selected_features','r')

for line in p:
	ls=line.split()
	selected_feature_names.append(int(ls[0]))
"""

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

print("Reached Combining")

####################Combining!##########################
for i in range(0,len(feature_set)):
	feature_set[i]=feature_set[i]+x_train_sg[i]

for i in range(0,len(feature_set_test)):
	feature_set_test[i]=feature_set_test[i]+x_test_sg[i]
#########################################################

print("GRADIENT BOOSTING CLASSIFIER")

from sklearn.ensemble import GradientBoostingClassifier


clf = GradientBoostingClassifier(random_state=20,learning_rate=0.1, n_estimators=1000,max_depth=3, min_samples_split=5, min_samples_leaf=1, subsample=1,max_features='sqrt')

clf.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf.score(feature_set, y_train))


print ("\nAccuracy on Testing Set :")
print (clf.score(feature_set_test, y_test))

y_pred=clf.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))
###############################################################
print("ADABOOST CLASSIFIER")
from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators=500,random_state=20)


clf.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf.score(feature_set, y_train))


print ("\nAccuracy on Testing Set :")
print (clf.score(feature_set_test, y_test))

y_pred=clf.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))
