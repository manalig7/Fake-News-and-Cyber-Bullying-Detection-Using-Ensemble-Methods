import pandas as pd  
import numpy as np
import sys
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

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

import spacy 
from textstat.textstat import easy_word_set, legacy_round 
from textstat.textstat import textstatistics
  
 ##########################################FUNCTIONS################################# 
def break_sentences(text): 
    nlp = spacy.load('en') 
    doc = nlp(text) 
    return doc.sents 
  
# Returns Number of Words in the text 
def word_count(text): 
    sentences = break_sentences(text) 
    words = 0
    for sentence in sentences: 
        words += len([token for token in sentence]) 
    return words 
  
# Returns the number of sentences in the text 
def sentence_count(text): 
    sentences = break_sentences(text) 
    return len(sentences) 

def num_syllables(word): 
    return textstatistics().syllable_count(word) 

# Returns the average number of syllables per 
# word in the text 
def avg_syllables_per_word(text): 
    syllable = num_syllables(text) 
    words = word_count(text) 
    ASPW = float(syllable) / float(words) 
    return legacy_round(ASPW, 1) 

##############################################FEATURES################################
#Number of Words

for i in range(0,len(x_train)):
	text=x_train[i]
	wordcount=word_count(text)
	feature_set[i].append(wordcount)

#Number of Sentences
for i in range(0,len(x_train)):
	text=x_train[i]
	sentencecount=sentence_count(text)
	feature_set[i].append(sentencecount)

#Average Number of Syllables per word
for i in range(0,len(x_train)):
	text=x_train[i]
	avg_syl=avg_syllables_per_word(text)
	feature_set[i].append(avg_syl)



"""
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

remove_list=[',','.','?',':','_','-','--','(',')','!','$']

for i in range(0,len(tokenized_text)):
	for item in remove_list:
		tokenized_text[i].remove(item)
	feature_set[i].append(len(tokenized_text))

"""
#3rd Feature- Grammatical Complexity
#Flesh-Kincaid Grade Level

