import pandas as pd  
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import pickle
import sys

tsv = 'FakeNewsNet_Dataset/fakenewsnet_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append(ls[0].lower())
	y_train.append(int(ls[1]))

tsv1 = 'FakeNewsNet_Dataset/fakenewsnet_test.txt'
f=open(tsv,'r')
x_test=[]
y_test=[]

for line in f:
	ls=line.split('\t')
	x_test.append(ls[0].lower())
	y_test.append(int(ls[1]))

print "Number of Training Samples"
print len(x_train)

pos=0
neg=0

for i in range(0,len(y_train)):
	if y_train[i]==0:
		neg=neg+1
	else:
		pos=pos+1

print "Fake News"
print pos
print "Real News"
print neg


from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
tvec.fit(x_train)

#filename = 'tfidf_model.txt'
#pickle.dump(tvec, open(filename, 'wb'))

x_train_tfidf = tvec.transform(x_train)
x_test_tfidf = tvec.transform(x_test)

print x_train_tfidf.shape



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(x_train_tfidf, y_train)

"""
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
"""

clf.fit(x_train_tfidf, y_train)

print "Accuracy on Training Set"
print clf.score(x_train_tfidf, y_train)
#print "Saving the Model"
#filename = 'text_model_rf.txt'
#pickle.dump(clf, open(filename, 'wb'))

print "Checking on Test Set"
print "Accuracy on Testing Set"
print clf.score(x_test_tfidf, y_test)

y_pred=clf.predict(x_test_tfidf)

print "Precision Score"
print precision_score(y_test, y_pred)
print "Recall Score"
print recall_score(y_test, y_pred)
print "F1 Score"
print f1_score(y_test, y_pred)




#######################################################################################
