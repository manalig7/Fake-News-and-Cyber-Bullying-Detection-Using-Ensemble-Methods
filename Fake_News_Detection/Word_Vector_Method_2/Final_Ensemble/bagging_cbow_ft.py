#Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

#Import the required packages

#Import pandas to read csv import pandas

#Import numpy for array related operations import numpy

#Import sklearn's feature selection algorithm
import numpy as np
np.random.seed(20)
from random import seed
seed(20)

from sklearn.feature_selection import SelectKBest
import pandas as pd  

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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import gensim
import pickle
from gensim.models import FastText


corpus = []
labels = []

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

tsv = '/Users/haritareddy/Documents/Method_2_Trial/Major-Project/Fake_News_Detection/Word_Vector_Method_2/finaldataset_train.txt'
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
tsv1 = '/Users/haritareddy/Documents/Method_2_Trial/Major-Project/Fake_News_Detection/Word_Vector_Method_2/finaldataset_test.txt'
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

model_FT = gensim.models.FastText.load("/Users/haritareddy/Documents/Method_2_Trial/Major-Project/Fake_News_Detection/Word_Vector_Method_2/cbow_ft.model")
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
for i in range(0,51):
	names.append(i)
#Create pandas data frame by loading the data from URL

dataframe = pd.read_csv("writeprints_selected_train.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
feature_set = array[:,0:50]
print('Length of feature set')
print(len(feature_set))
print('Length of feature set sg')
print(len(x_train_sg))

dataframe_test = pd.read_csv("writeprints_selected_test.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe_test.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
feature_set_test = array[:,0:50]


feature_set=feature_set.tolist()
feature_set_test=feature_set_test.tolist()



######################################

n_writeprints=10
size_writeprints=700
print (n_writeprints, size_writeprints)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
clf = RandomForestClassifier(random_state=20)
clf1 = LogisticRegression(random_state=20,solver='lbfgs')
predictions=[]
for p in range(0,len(feature_set_test)):
	predictions.append([])

#Need to Select different subsets
for i in range(0,n_writeprints):
	temp_list=[]
	temp_list_y=[]
	temp_list_sg=[]
	for j in range(0,size_writeprints):
		num=random.randint(0,len(feature_set)-1)
		temp_list.append(feature_set[num])
		temp_list_sg.append(x_train_sg[num])
		temp_list_y.append(y_train[num])
	clf.fit(temp_list,temp_list_y)
	clf1.fit(temp_list_sg,temp_list_y)
	pred=clf.predict(feature_set_test)
	pred_sg=clf1.predict(x_test_sg)
	for p in range(0,len(pred)):
		predictions[p].append(pred[p])
	for p in range(0,len(pred_sg)):
		predictions[p].append(pred_sg[p])

for i in range(0,len(predictions)):
	for j in range(0,len(predictions[i])):
		predictions[i][j]=int(predictions[i][j])

final_pred=[]

for i in range(0,len(predictions)):
	#print predictions[i]
	final_pred.append(max(set(predictions[i]), key=predictions[i].count))

print("Accuracy on Testing Set")
acc=0
total=0
for i in range(0,len(y_test)):
	if y_test[i]==final_pred[i]:
		acc=acc+1
	total=total+1

accuracy=float(acc)/float(total)
print(accuracy)
print ("\nPrecision Score")
print (precision_score(y_test, final_pred))
print ("\nRecall Score")
print (recall_score(y_test, final_pred))
print ("\nF1 Score")
print (f1_score(y_test, final_pred))

######################################	
######################################

n_writeprints=15
size_writeprints=700
print (n_writeprints, size_writeprints)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
clf = RandomForestClassifier(random_state=20)
clf1 = LogisticRegression(random_state=20,solver='lbfgs')
predictions=[]
for p in range(0,len(feature_set_test)):
	predictions.append([])

#Need to Select different subsets
for i in range(0,n_writeprints):
	temp_list=[]
	temp_list_y=[]
	temp_list_sg=[]
	for j in range(0,size_writeprints):
		num=random.randint(0,len(feature_set)-1)
		temp_list.append(feature_set[num])
		temp_list_sg.append(x_train_sg[num])
		temp_list_y.append(y_train[num])
	clf.fit(temp_list,temp_list_y)
	clf1.fit(temp_list_sg,temp_list_y)
	pred=clf.predict(feature_set_test)
	pred_sg=clf1.predict(x_test_sg)
	for p in range(0,len(pred)):
		predictions[p].append(pred[p])
	for p in range(0,len(pred_sg)):
		predictions[p].append(pred_sg[p])

for i in range(0,len(predictions)):
	for j in range(0,len(predictions[i])):
		predictions[i][j]=int(predictions[i][j])

final_pred=[]

for i in range(0,len(predictions)):
	#print predictions[i]
	final_pred.append(max(set(predictions[i]), key=predictions[i].count))

print("Accuracy on Testing Set")
acc=0
total=0
for i in range(0,len(y_test)):
	if y_test[i]==final_pred[i]:
		acc=acc+1
	total=total+1

accuracy=float(acc)/float(total)
print(accuracy)
print ("\nPrecision Score")
print (precision_score(y_test, final_pred))
print ("\nRecall Score")
print (recall_score(y_test, final_pred))
print ("\nF1 Score")
print (f1_score(y_test, final_pred))

######################################	
##################################
"""
n_writeprints=8
size_writeprints=700
print n_writeprints, size_writeprints

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
clf = RandomForestClassifier(random_state=seed)
clf1 = LogisticRegression(random_state=seed, solver='lbfgs')
predictions=[]
for p in range(0,len(feature_set_test)):
	predictions.append([])

#Need to Select different subsets
for i in range(0,n_writeprints):
	temp_list=[]
	temp_list_y=[]
	temp_list_sg=[]
	for j in range(0,size_writeprints):
		num=random.randint(0,len(feature_set)-1)
		temp_list.append(feature_set[num])
		temp_list_sg.append(x_train_sg[num])
		temp_list_y.append(y_train[num])
	clf.fit(temp_list,temp_list_y)
	clf1.fit(temp_list_sg,temp_list_y)
	pred=clf.predict(feature_set_test)
	pred_sg=clf1.predict(x_test_sg)
	for p in range(0,len(pred)):
		predictions[p].append(pred[p])
	for p in range(0,len(pred_sg)):
		predictions[p].append(pred_sg[p])

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

##################################

n_writeprints=12
size_writeprints=600
print n_writeprints, size_writeprints

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
clf = RandomForestClassifier(random_state=seed)
clf1 = LogisticRegression(random_state=seed, solver='lbfgs')
predictions=[]
for p in range(0,len(feature_set_test)):
	predictions.append([])

#Need to Select different subsets
for i in range(0,n_writeprints):
	temp_list=[]
	temp_list_y=[]
	temp_list_sg=[]
	for j in range(0,size_writeprints):
		num=random.randint(0,len(feature_set)-1)
		temp_list.append(feature_set[num])
		temp_list_sg.append(x_train_sg[num])
		temp_list_y.append(y_train[num])
	clf.fit(temp_list,temp_list_y)
	clf1.fit(temp_list_sg,temp_list_y)
	pred=clf.predict(feature_set_test)
	pred_sg=clf1.predict(x_test_sg)
	for p in range(0,len(pred)):
		predictions[p].append(pred[p])
	for p in range(0,len(pred_sg)):
		predictions[p].append(pred_sg[p])

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

######################################

n_writeprints=30
size_writeprints=500
print n_writeprints, size_writeprints

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
clf = RandomForestClassifier(random_state=seed)
clf1 = LogisticRegression(random_state=seed, solver='lbfgs')
predictions=[]
for p in range(0,len(feature_set_test)):
	predictions.append([])

#Need to Select different subsets
for i in range(0,n_writeprints):
	temp_list=[]
	temp_list_y=[]
	temp_list_sg=[]
	for j in range(0,size_writeprints):
		num=random.randint(0,len(feature_set)-1)
		temp_list.append(feature_set[num])
		temp_list_sg.append(x_train_sg[num])
		temp_list_y.append(y_train[num])
	clf.fit(temp_list,temp_list_y)
	clf1.fit(temp_list_sg,temp_list_y)
	pred=clf.predict(feature_set_test)
	pred_sg=clf1.predict(x_test_sg)
	for p in range(0,len(pred)):
		predictions[p].append(pred[p])
	for p in range(0,len(pred_sg)):
		predictions[p].append(pred_sg[p])

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

######################################
"""
