import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec


def fit_transform(x,d):
	res=[]
	#print(pad_len)
	for i in range(0,len(x)):
		temp=[]
		for word in d[i]:
			#word=word.encode('ascii','ignore')
			temp.append(np.mean(model_W2V.wv[word]))
		if len(temp)<pad_len:
			temp.extend([0]*(pad_len-len(temp)))		
		res.append(temp)

	return res


x = []
##### training dataset #####

tsv = 'dataset/finaldataset_train.txt'
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
tsv1 = 'dataset/finaldataset_test.txt'
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

model_W2V = Word2Vec(data, size=10, window=5, min_count=1, workers=5, sg=0)

print "CBOW W2V model_done!"

#model_W2V.train(data1)

XVAL=fit_transform(x,data)
x_train=[]
x_train=XVAL[:m]
#print(np.array(x_train).shape)
x_test = []
x_test=XVAL[m:]
#print(np.array(x_test).shape)



################# Naive Bayes Classifier ####################

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score

clf = MultinomialNB()
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

