import keras.backend as K
import multiprocessing
import tensorflow as tf
import numpy as np
from gensim.models.word2vec import Word2Vec

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import io
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam



# Set random seed (for reproducibility)
np.random.seed(1001)


dataset_location = 'FakeNewsNet_Dataset/fakenewsnet_train.txt.csv'
model_location = 'word2vec_model.txt'

corpus = []
labels = []

# Parse tweets and sentiments
with io.open(dataset_location, 'r', encoding='utf-8') as df:
    for i, line in enumerate(df):
        if i == 0:
            # Skip the header
            continue

        parts = line.strip().split('\t')
        
        # Sentiment (0 = Negative, 1 = Positive)
        labels.append(int(parts[1].strip()))
        
        # Tweet
        news= parts[0].strip()
        if tweet.startswith('"'):
            news = news[1:]
        if tweet.endswith('"'):
            news = news[::-1]
        
        corpus.append(news.strip().lower())
        
print('Corpus size: {}'.format(len(corpus)))

# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, news in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(news) if not t.startswith('@')]
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