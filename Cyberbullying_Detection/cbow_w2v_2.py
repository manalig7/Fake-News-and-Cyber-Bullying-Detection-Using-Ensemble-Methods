import numpy as np
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

tsv = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_clean_train.txt'
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
tsv1 = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_clean_test.txt'
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

#model_W2V = Word2Vec(data, size=10, window=5, min_count=1, workers=5, sg=0,max_vocab_size=10000)
#model_W2V.save("CBOW_W2V.model")
model_W2V = Word2Vec.load("CBOW_W2V.model")

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

"""
print ("################### SVM Classifier ###############")

from sklearn.svm import SVC

clf = SVC(gamma='auto')
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
"""

