import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import FastText
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import pandas as pd


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

tsv = '/home/user/Documents/Major Project/Implement-CB/DS_spellchecked/finaldb_train_spellcorrect_segment.txt'
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
tsv1 = '/home/user/Documents/Major Project/Implement-CB/DS_spellchecked/finaldb_test_spellcorrect_segment.txt'
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
x_train=pd.DataFrame(x_train)
#print(np.array(x_train).shape)
x_test = []
x_test=XVAL[m:]
x_test=pd.DataFrame(x_test)
print(x_test.shape)



print (" ################## MLP #####################")

import tensorflow
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=3)

ip=x_train.shape[1]
model = Sequential()
model.add(Dense((ip/5), input_dim=(ip), activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense((ip/100), activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adagrad',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25,batch_size=(x_train.shape[0]/100), callbacks=[early_stopping_monitor])

print(model.metrics_names)

print "\nAccuracy on Training Set :"
score = model.evaluate(x_train, y_train, batch_size=(x_train.shape[0]/100))
print (score)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
score = model.evaluate(x_test, y_test, batch_size=(x_train.shape[0]/100))
print (score)

y_pred = model.predict_classes(x_test)
#print y_pred

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

