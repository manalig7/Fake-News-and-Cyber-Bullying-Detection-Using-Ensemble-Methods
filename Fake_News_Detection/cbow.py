import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from gensim.models import FastText


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
print(data)

model_ted = Word2Vec(data, size=100, window=5, min_count=1, workers=5, sg=0)


model_ted = FastText(data, size=100, window=5, min_count=1, workers=5,sg=0)

#def fit_transform(X):
	
