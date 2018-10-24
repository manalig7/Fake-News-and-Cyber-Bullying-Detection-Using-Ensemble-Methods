import keras.backend as K
import multiprocessing
import tensorflow as tf
import numpy as np
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

# Set random seed (for reproducibility)
np.random.seed(1000)


dataset_location = 'final_cleaned_train1.csv'
model_location = 'model.txt'

corpus = []
labels = []

# Parse tweets and sentiments
with io.open(dataset_location, 'r', encoding='utf-8') as df:
    for i, line in enumerate(df):
        if i == 0:
            # Skip the header
            continue

        parts = line.strip().split(',')
        
        # Sentiment (0 = Negative, 1 = Positive)
        labels.append(int(parts[1].strip()))
        
        # Tweet
        tweet = parts[2].strip()
        if tweet.startswith('"'):
            tweet = tweet[1:]
        if tweet.endswith('"'):
            tweet = tweet[::-1]
        
        corpus.append(tweet.strip().lower())
        
print('Corpus size: {}'.format(len(corpus)))

# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
    tokenized_corpus.append(tokens)
    
# Gensim Word2Vec model
vector_size = 120
window_size = 8

# Create Word2Vec
word2vec = Word2Vec(sentences=tokenized_corpus,
                    size=vector_size, 
                    window=window_size, 
                    negative=20,
                    iter=40,
                    seed=1000,
                    workers=multiprocessing.cpu_count(),sg=1)

# Copy word vectors and delete Word2Vec model  and original corpus to save memory
X_vecs = word2vec.wv
del word2vec
del corpus

# Train subset size (0 < size < len(tokenized_corpus))
train_size = int(0.95*len(tokenized_corpus))

# Test subset size (0 < size < len(tokenized_corpus) - train_size)
test_size = int(0.05*len(tokenized_corpus))

# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))
    
print('Average tweet length: {}'.format(avg_length / float(len(tokenized_corpus))))
print('Max tweet length: {}'.format(max_length))

# Tweet max length (number of tokens)
max_tweet_length = max_length

# Create train and test sets
# Generate random indexes
indexes = set(np.random.choice(len(tokenized_corpus), train_size + test_size, replace=False))

X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_train = np.zeros((train_size, 2), dtype=np.int32)
X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
Y_test = np.zeros((test_size, 2), dtype=np.int32)

for i, index in enumerate(indexes):
    for t, token in enumerate(tokenized_corpus[index]):
        if t >= max_tweet_length:
            break
        
        if token not in X_vecs:
            continue
    
        if i < train_size:
            X_train[i, t, :] = X_vecs[token]
        else:
            X_test[i - train_size, t, :] = X_vecs[token]
            
    if i < train_size:
        Y_train[i, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
    else:
        Y_test[i - train_size, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
        
# Keras convolutional model
batch_size = 100
nb_epochs = 100

model = Sequential()

model.add(Conv1D(300, kernel_size=5, activation='relu', padding='same', input_shape=(max_tweet_length, vector_size)))

"""
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
"""
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=nb_epochs,
          validation_data=(X_test, Y_test),
callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])

y_pred=model.predict(X_test)

y_res=y_pred
for i in range(0,len(y_res)):
        y_res[i][0]=round(y_res[i][0])
        y_res[i][1]=round(y_res[i][1])

ac_count=0
tot_count=0

for i in range(0,len(Y_test)):
    if Y_test[i][0]==y_res[i][0] and Y_test[i][1]==y_res[i][1]:
        ac_count=ac_count+1
    tot_count=tot_count+1


print "Accuracy"

print (float(ac_count)/float(tot_count))*100


print "Weighted Precision Score"
print precision_score(Y_test, y_res, average='weighted')
print "Weighted Recall Score"
print recall_score(Y_test, y_res, average='weighted')
print "Weighted F1 Score"
print f1_score(Y_test, y_res, average='weighted')

print "Micro Precision Score"
print precision_score(Y_test, y_res, average='micro')
print "Micro Recall Score"
print recall_score(Y_test, y_res, average='micro')
print "Micro F1 Score"
print f1_score(Y_test, y_res, average='micro')

print "Macro Precision Score"
print precision_score(Y_test, y_res, average='macro')
print "Macro Recall Score"
print recall_score(Y_test, y_res, average='macro')
print "Macro F1 Score"
print f1_score(Y_test, y_res, average='macro')