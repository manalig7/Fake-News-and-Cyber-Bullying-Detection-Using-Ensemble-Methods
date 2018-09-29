import pandas as pd  
import numpy as np
import sys
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

tsv = 'FakeNewsNet_Dataset/fakenewsnet_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append(ls[0].lower())
	y_train.append(int(ls[1]))

feature_set=[]

for i in range(0,len(x_train)):
	feature_set.append([])

#x_train contains the list of pieces of news texts
#1st feature- Quantity (number of syllables, number of words, number of sentences)

#Number of Sentences
for i in range(0,len(x_train)):
	text=x_train[i]
	num_fullstops=text.count('.')
	if num_fullstops==0:
		num_fullstops=1
	num_sentences=num_fullstops
	feature_set[i].append(num_sentences)


#Number of Words
#For number of words, we convert the pieces of texts into tokenized form and remove special characters like . and ,, i.e we consider only proper words
# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_text = []

for i, news in enumerate(x_train):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(news) if not t.startswith('@')]
    tokenized_text.append(tokens)

for i in range(0,len(tokenized_text)):
	tokenized_text[i].remove(',','.','?',':','_','-','--','(',')','!','$')
	feature_set[i].append(len(tokenized_text))
