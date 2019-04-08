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

tsv = '/Users/haritareddy/Desktop/Major-Project-Final/Major-Project/Cyberbullying_Detection/Spell_Check_and_Segmentation/finalcb_clean_train_py_spell_check_segment_all.txt'
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
tsv1 = '/Users/haritareddy/Desktop/Major-Project-Final/Major-Project/Cyberbullying_Detection/Spell_Check_and_Segmentation/finalcb_clean_test_py_spell_check_segment_all.txt'
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

#model_FT = FastText(data, size=10, window=5, min_count=1, workers=5, sg=1,max_vocab_size=10000)
#model_FT.save("SG_fasttext.model")
model_FT = FastText.load("SG_fasttext.model")

print "SG FT model_done!"


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

import tensorflow
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=3)
x_train=np.array(x_train)
x_test=np.array(x_test)
ip=x_train.shape[1]
model = Sequential()
model.add(Dense((ip/5), input_dim=(ip), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense((ip/100), activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='mean_squared_error', optimizer='adagrad',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25,batch_size=(x_train.shape[0]/100), callbacks=[early_stopping_monitor])

print(model.metrics_names)

print "\nAccuracy on Training Set :"
score = model.evaluate(x_train, y_train, batch_size=(x_train.shape[0]/100))
print (score)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
score = model.evaluate(x_test, y_test, batch_size=(x_test.shape[0]/100))
print (score)

y_pred = model.predict_classes(x_test)
#print y_pred

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


"""

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


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier,  VotingClassifier


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

print ("################### Ensemble Bagging Classifier ###############")

clf4 = BaggingClassifier(GaussianNB(),max_samples=0.4, max_features=0.8,n_estimators=300,random_state=20)

clf4 = clf4.fit(x_train, y_train)

print "\nAccuracy on Training Set :"
print clf4.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf4.score(x_test, y_test)

y_pred=clf4.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)



print ("################### Ensemble Boosting Classifier ###############")


clf5 = AdaBoostClassifier(n_estimators=300,random_state=20)

clf5 = clf5.fit(x_train, y_train)

print "\nAccuracy on Training Set :"
print clf5.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf5.score(x_test, y_test)

y_pred=clf5.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

print ("################### SVM Classifier ###############")

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=20, tol=1e-5)
clf = clf.fit(x_train, y_train)

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
