import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
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
				temp.append(np.mean(model_W2V.wv[voc[j]]))
			else :
				temp.append(0)		
		res.append(temp)
	return res

x = []
##### training dataset #####

tsv = 'augmented_train.tsv'
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
tsv1 = 'finalcb_clean_test_py_spell_check_segment_all.txt'
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
model_W2V = Word2Vec.load("SG_W2V_augmented.model")

print "SG W2V model_done!"

voc=list(model_W2V.wv.vocab)
print(len(voc))

XVAL=fit_transform(data)

print ("Transformed!!")

#print(m)

#print (XVAL)

x_train=[]
x_train=np.array(XVAL[:m])
ip1= (x_train).shape[1]
ip0= (x_train).shape[0]
print(ip1)

x_test = []
x_test=np.array(XVAL[m:])
ip11= (x_test).shape[1]
ip10= (x_test).shape[0]

#x_train = x_train[:,:,None]
#x_test = x_test[:,:,None]

#y_train = y_train[:,:,None]
#y_test = y_test[:,:,None]

print (x_train.shape)
print(x_test.shape)



print (" ################## LSTM #####################")

import tensorflow as tf
old_v=tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Embedding

#timesteps=1
#ip0=x_train.shape[1]
ip1=x_train.shape[1]
model=Sequential()
model.add(Embedding(len(voc),5,input_length = ip1))
model.add(LSTM((ip1/10), activation='relu'))
#model.add(LSTM(units=6, return_sequences=True))
#model.add(LSTM(units=6, return_sequences=True))
#model.add(LSTM(units=18, return_sequences=True, name='output', activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=5)

print(model.metrics_names)

print "\nAccuracy on Training Set :"
score = model.evaluate(x_train, y_train, batch_size=1000)#(x_train.shape[0]/100))
print (score)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
score = model.evaluate(x_test, y_test, batch_size=25)#(x_train.shape[0]/100))
print (score)

y_pred = model.predict_classes(x_test)
#print y_pred

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)