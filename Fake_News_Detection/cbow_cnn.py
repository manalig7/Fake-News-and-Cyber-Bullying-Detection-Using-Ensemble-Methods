import pandas as pd  
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
from tqdm import tqdm
import sys
from keras.models import model_from_json
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

ps = PorterStemmer()

def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

csv = 'Finaldataset/finaldataset_train.txt'
my_df = pd.read_csv(csv,sep='\t',index_col=None,header=None)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
#print my_df[:0]
x_train = my_df.iloc[:,0]
y_train = my_df.iloc[:,1]


csv = 'Finaldataset/finaldataset_test.txt'
my_df = pd.read_csv(csv,sep='\t',index_col=None,header=None)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
x_test = my_df.iloc[:,0]
y_test = my_df.iloc[:,1]



"""
all_x = pd.concat([x_train,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')

cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha
model_ug_cbow.save('w2v_model_ug_cbow_stem.word2vec')
"""

from gensim.models import KeyedVectors
model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')

embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = model_ug_cbow.wv[w]



tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)




length = []
for x in x_train:
    length.append(len(x.split()))

x_train_seq = pad_sequences(sequences, maxlen=max(length))




num_words = 100000
embedding_matrix = np.zeros((num_words, 100))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

seed = 7



from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

model_ptw2v = Sequential()
e = Embedding(100000, 100, weights=[embedding_matrix], input_length=max(length), trainable=True)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train, validation_split=0.20, epochs=4, batch_size=800, verbose=1)

# serialize model to JSON
model_json = model_ptw2v.to_json()
with open("model_cbow.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_cbow.h5")
print("Saved model to disk")


json_file = open('model_cbow.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_cbow.h5")
print("Loaded model from disk")


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
y_pred=model_ptw2v.predict_on_batch(x_test)
print "Prediction"
print y_pred[0]
