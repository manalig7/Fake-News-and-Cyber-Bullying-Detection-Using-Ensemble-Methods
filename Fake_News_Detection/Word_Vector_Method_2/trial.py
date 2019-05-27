import keras.backend as K
import multiprocessing
import numpy as np
np.random.seed(20)
from random import seed
seed(20)
import tensorflow as tf
tf.set_random_seed(20)
import sys
from gensim.models.word2vec import Word2Vec

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import io
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import gensim
from gensim.models import Word2Vec
# Set random seed (for reproducibility)




corpus = []
labels = []

def fit_transform(d):
    res=[]
    for i in range(0,len(d)):
        temp=[] 
        for j in range(0,len(voc)):
            #print(voc[j])
            if voc[j] in d[i]:
                #print  (np.mean(model_W2V.wv[voc[j]])) 
                temp.append(np.mean(model_W2V.wv[voc[j]]))
            else :
                temp.append(0)      
        res.append(temp)
    return res

tsv = 'finaldataset_train.txt'
f=open(tsv,'r')
x = []
Y_train=[]
tokenized_corpus=[]
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
    tokenized_corpus.append(temp)
    lent.append(len(temp)) 
    Y_train.append(int(ls[1]))
f.close()

model_W2V = gensim.models.Word2Vec.load("sg_w2v.model")

voc=list(model_W2V.wv.vocab)

# Copy word vectors and delete Word2Vec model  and original corpus to save memory
X_vecs = model_W2V.wv



# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
##################################READING IN THE TEST SET###################################

tsv = 'finaldataset_test.txt'
f=open(tsv,'r')
x = []
Y_test=[]
tokenized_corpus_test=[]
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
    tokenized_corpus_test.append(temp)
    lent.append(len(temp)) 
    Y_test.append(int(ls[1]))
f.close()

print ("Reached Here")

# Tweet max length (number of tokens)
max_tweet_length = max_length

# Create train and test sets
# Generate random indexes
#indexes = set(np.random.choice(len(tokenized_corpus), train_size + test_size, replace=False))


sys.exit()
X_train=fit_transform(tokenized_corpus)
print ("Done transforming train")
X_test=fit_transform(tokenized_corpus_test)
print ("Done transforming test")

print ("Length of train: ", len(X_train))
print ("Length of test: ", len(X_test))

print (X_train[0])
print("############")
print (X_test[0])



