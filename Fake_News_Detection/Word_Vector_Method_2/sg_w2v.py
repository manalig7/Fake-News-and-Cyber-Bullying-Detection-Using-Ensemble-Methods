import keras.backend as K
import multiprocessing
import numpy as np
np.random.seed(20)
from random import seed
seed(20)
import tensorflow as tf
tf.set_random_seed(20)

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
"""
# Gensim Word2Vec model
vector_size = 120
window_size = 8

# Create Word2Vec

model_W2V  = Word2Vec(sentences=tokenized_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=40,
                    seed=1000,
                    workers=multiprocessing.cpu_count(),sg=1)
"""

model_W2V =Word2Vec(tokenized_corpus, size=10, window=5, min_count=1, workers=5, sg=1,max_vocab_size=10000)


model_W2V.save("sg_w2v.model")


#model_W2V = gensim.models.Word2Vec.load("sg_w2v.model")
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
############################################################################################    
print ("Reached Here")

# Tweet max length (number of tokens)
max_tweet_length = max_length

# Create train and test sets
# Generate random indexes
#indexes = set(np.random.choice(len(tokenized_corpus), train_size + test_size, replace=False))



X_train=fit_transform(tokenized_corpus)
print ("Done transforming train")
X_test=fit_transform(tokenized_corpus_test)
print ("Done transforming test")
#Y_train=labels
#Y_test=labels_test

print ("Finished with test")


"""
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs')
"""

"""
from sklearn.ensemble import GradientBoostingClassifier


clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000,max_depth=3, min_samples_split=5, min_samples_leaf=1, subsample=1,max_features='sqrt')


clf.fit(X_train,Y_train)
print "\nAccuracy on Training Set :"
print clf.score(X_train, Y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(X_test, Y_test)

y_pred=clf.predict(X_test)

print "\nPrecision Score"
print precision_score(Y_test, y_pred)
print "\nRecall Score"
print recall_score(Y_test, y_pred)
print "\nF1 Score"
print f1_score(Y_test, y_pred)
"""
print("################# Naive Bayes Classifier ####################")

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train,Y_train)

print ("\nAccuracy on Training Set :")
print (clf.score(X_train, Y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf.score(X_test, Y_test))

y_pred=clf.predict(X_test)

print ("\nPrecision Score")
print (precision_score(Y_test, y_pred))
print ("\nRecall Score")
print (recall_score(Y_test, y_pred))
print ("\nF1 Score")
print (f1_score(Y_test, y_pred))


print ("################### Random Forest Classifier ###############")

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier()
clf.fit(X_train,Y_train)

print ("\nAccuracy on Training Set :")
print (clf.score(X_train, Y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf.score(X_test, Y_test))

y_pred=clf.predict(X_test)

print ("\nPrecision Score")
print (precision_score(Y_test, y_pred))
print ("\nRecall Score")
print (recall_score(Y_test, y_pred))
print ("\nF1 Score")
print (f1_score(Y_test, y_pred))

print ("################### Logistic regression Classifier ###############")

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs')

clf.fit(X_train,Y_train)

print ("\nAccuracy on Training Set :")
print (clf.score(X_train, Y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf.score(X_test, Y_test))

y_pred=clf.predict(X_test)

print ("\nPrecision Score")
print (precision_score(Y_test, y_pred))
print ("\nRecall Score")
print (recall_score(Y_test, y_pred))
print ("\nF1 Score")
print (f1_score(Y_test, y_pred))
#print f1_score(y_test, y_pred)