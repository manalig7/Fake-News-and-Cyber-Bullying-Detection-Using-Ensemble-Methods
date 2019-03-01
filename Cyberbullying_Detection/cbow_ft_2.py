import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
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

tsv = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_clean_train.txt'
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
	lent.append(len(temp))  
	y_test.append(int(ls[1]))
f.close()

pad_len=max(lent)

#model_FT = FastText(data, size=10, window=5, min_count=1, workers=5, sg=0,max_vocab_size=10000)
#model_FT.save("CBOW_fasttext.model")

model_FT = FastText.load("CBOW_fasttext.model")

print "CBOW FT model_done!"

voc=list(model_FT.wv.vocab)
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
"""
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
"""

print ("################### Random Forest Classifier ###############")

from sklearn.ensemble import RandomForestClassifier

"""
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
"""
print ("################### Logistic regression Classifier ###############")

from sklearn.linear_model import LogisticRegression
"""
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

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


print ("################### Ensemble Voting Classifier ###############")

clf1 = GaussianNB()
clf2 = LogisticRegression(random_state=20, solver='lbfgs')
clf3 = RandomForestClassifier(random_state=20)
clf4 = BaggingClassifier(GaussianNB(),max_samples=0.4, max_features=0.8,n_estimators=300,random_state=20)
clf5 = AdaBoostClassifier(n_estimators=300,random_state=20)

eclf1 = VotingClassifier(estimators=[('gb', clf1), ('lr', clf2), ('rf', clf3)], voting='hard')
# ('gb', clf1), ('lr', clf2), ('rf', clf3) ,('ada', clf5), ('bag', clf4)
eclf1 = eclf1.fit(x_train, y_train)

print "\nAccuracy on Training Set :"
print eclf1.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print eclf1.score(x_test, y_test)

y_pred=eclf1.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


print ("################### Ensemble Boosting Classifier ###############")


clf5 = AdaBoostClassifier(n_estimators=300,random_state=20)

clf5 = VotingClassifier(estimators=[('gb', clf1), ('lr', clf2), ('rf', clf3)], voting='hard')
# ('gb', clf1), ('lr', clf2), ('rf', clf3) ,('ada', clf5), ('bag', clf4)
clf5 = clf5.fit(x_train, y_train)

print "\nAccuracy on Training Set :"
print clf5.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf5.score(x_test, y_test)

y_pred=clf5.predict(final_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


