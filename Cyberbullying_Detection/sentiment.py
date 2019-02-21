from textblob import TextBlob
import pandas as pd  
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import pickle
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer


"""

	testimonial = TextBlob(ls[0].decode('utf-8'))
	print testimonial.sentiment
	print ls[1]
"""

analyser = SentimentIntensityAnalyzer()
tsv = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_clean_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	reslist=[]
	ls=line.split('\t')
	testimonial = TextBlob(ls[0].decode('utf-8'))
	res= testimonial.sentiment
	reslist.extend([res[0],res[1]])
	#print reslist
	score = analyser.polarity_scores(ls[0])
	## merge this reslist to feature_vector in Writeprints
	reslist.extend(score.values())
	#print "P="+str(res[0])
	#print "S="+str(res[1])
	x_train.append(reslist)
	y_train.append(int(ls[1]))
f.close()

#print x_train
m=len(x_train)


tsv1 = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_clean_test.txt'
f=open(tsv1,'r')

x_test=[]
y_test=[]

for line in f:
	reslist=[]
	ls=line.split('\t')
	testimonial = TextBlob(ls[0].decode('utf-8'))
	res= testimonial.sentiment
	reslist.extend([res[0],res[1]])
	#print reslist
	score = analyser.polarity_scores(ls[0])
	## merge this reslist to feature_vector
	reslist.extend(score.values())
	#print "P="+str(res[0])
	#print "S="+str(res[1])
	x_test.append(reslist)
	y_test.append(int(ls[1]))
f.close()

#print x_train

print("################# Naive Bayes Classifier ####################")

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(x_train,y_train)

print "\nAccuracy on Training Set :"
print clf.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(x_test, y_test)

y_pred=clf.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


print ("################### Random Forest Classifier ###############")

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(random_state=20)
clf.fit(x_train,y_train)

print "\nAccuracy on Training Set :"
print clf.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(x_test, y_test)

y_pred=clf.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

print ("################### Logistic regression Classifier ###############")

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=20, solver='lbfgs')

clf.fit(x_train,y_train)
print "\nAccuracy on Training Set :"
print clf.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(x_test, y_test)

y_pred=clf.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

print ("################### SVM Classifier ###############")

from sklearn.svm import SVC

clf = SVC(gamma='auto',random_state=20)
clf.fit(x_train,y_train)
print "\nAccuracy on Training Set :"
print clf.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(x_test, y_test)

y_pred=clf.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)
