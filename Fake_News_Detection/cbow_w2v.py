import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec


def fit_transform(x,d):
	res=[]
	for i in range(0,len(x_train)):
		temp=[]
		for word in d[i]:
			#word=word.encode('ascii','ignore')
			temp.append(np.mean(model_W2V.wv[word]))		
		res.append(temp)
	return res


##### training dataset #####

tsv = 'dataset/finaldataset_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]
data=[]

tokenizer = RegexpTokenizer(' ', gaps=True)

for line in f :
	ls=line.split('\t')
	x_train.append(ls[0])
	temp = [] 
	#print(ls[0])
	for j in tokenizer.tokenize(ls[0].decode('utf-8')):
		#print(j) 
	       	temp.append(j) 
	data.append(temp) 
	y_train.append(int(ls[1]))
f.close()
print(len(x_train))

model_W2V = Word2Vec(data, size=10, window=5, min_count=1, workers=5, sg=0)

print "model_done!"

X_train=fit_transform(x_train,data)

#print(np.asarray(X_train))


##### testing dataset #####
tsv1 = 'dataset/finaldataset_test.txt'
f=open(tsv1,'r')
x_test=[]
y_test=[]

data1=[]
for line in f:
	ls=line.split('\t')
	x_test.append(ls[0])
	temp = [] 
	#print(ls[0])
	for j in tokenizer.tokenize(ls[0].decode('utf-8')):
		#print(j) 
	       	temp.append(j) 
	data1.append(temp) 
	y_test.append(int(ls[1]))
f.close()

model_W2V.train(data1)

X_test=fit_transform(x_test,data1)
