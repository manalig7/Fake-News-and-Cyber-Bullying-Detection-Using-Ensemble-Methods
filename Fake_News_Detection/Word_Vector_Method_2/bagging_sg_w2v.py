#Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

#Import the required packages

#Import pandas to read csv import pandas

#Import numpy for array related operations import numpy

#Import sklearn's feature selection algorithm
import numpy as np
from random import seed
seed(20)
np.random.seed(20)
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



corpus = []
labels = []

def fit_transform(d):
    res=[]
    for i in range(0,len(d)):
        temp=[] 
        for j in range(0,len(voc)):
            #print(voc[j])
            if voc[j] in d[i]:
                #print  (np.mean(model_W2V.wv[voc[j]])) 
                temp.append(np.mean(model_W2V.wv[voc[j]]))
            else :
                temp.append(0)      
        res.append(temp)
    return res

tsv = 'finaldataset_train.txt'
f=open(tsv,'r')
x = []
y_train=[]
tokenized_corpus=[]
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
    tokenized_corpus.append(temp)
    lent.append(len(temp)) 
    y_train.append(int(ls[1]))
f.close()
"""  
# Gensim Word2Vec model
vector_size = 120
window_size = 8

# Create Word2Vec

model_W2V  = Word2Vec(sentences=tokenized_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=40,
                    seed=1000,
                    workers=multiprocessing.cpu_count(),sg=1)
"""


model_W2V = gensim.models.Word2Vec.load("sg_w2v.model")
voc=list(model_W2V.wv.vocab)

X_vecs = model_W2V.wv



# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
##################################READING IN THE TEST SET###################################

tsv = 'finaldataset_test.txt'
f=open(tsv,'r')
x = []
Y_test=[]
tokenized_corpus_test=[]
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
    tokenized_corpus_test.append(temp)
    lent.append(len(temp)) 
    y_test.append(int(ls[1]))
f.close()

print ("Reached Here")

# Tweet max length (number of tokens)
max_tweet_length = max_length

# Create train and test sets
# Generate random indexes
#indexes = set(np.random.choice(len(tokenized_corpus), train_size + test_size, replace=False))



x_train_sg=fit_transform(tokenized_corpus)
print ("Done transforming train")
x_test_sg=fit_transform(tokenized_corpus_test)
print ("Done transforming test")
#Y_train=labels
#Y_test=labels_test

print ("Finished with test")
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
model = LogisticRegression()
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
clf = RandomForestClassifier()
clf1 = LogisticRegression(solver='lbfgs')
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

print ("Accuracy on Testing Set")
acc=0
total=0
for i in range(0,len(y_test)):
	if y_test[i]==final_pred[i]:
		acc=acc+1
	total=total+1

accuracy=float(acc)/float(total)
print (accuracy)
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
clf = RandomForestClassifier()
clf1 = LogisticRegression(solver='lbfgs')
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

print ("Accuracy on Testing Set")
acc=0
total=0
for i in range(0,len(y_test)):
	if y_test[i]==final_pred[i]:
		acc=acc+1
	total=total+1

accuracy=float(acc)/float(total)
print (accuracy)
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
