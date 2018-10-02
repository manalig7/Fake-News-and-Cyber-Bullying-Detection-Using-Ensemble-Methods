"""
REQUIRED FEATURES
1. Quantity: 
Number of Syllables -
Number of Words -
Number of Sentences -
2. Vocabulary complexity: 
Number of big words - 
Number of syllables per word -
3. Grammmatical Complexity
Number of short sentences-
Number of long sentences-
Flesch-Kincaid -
Average Number of Words per sentence -
Sentence Complexity
Number of Conjunctions
4. Uncertainity
Number of words expressing uncertainity
Number of tentative words
Modal Verbs-
5. Specificity and Expressiveness
Rate of adjectives and adverbs
Number of affective terms
6. Verbal Non-Immediacy
Self References
Number of first, second and third person pronoun usage

"""


import pandas as pd  
import numpy as np
import sys
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

short_sentence_cutoff=10

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

#Average sentence length
def avg_sentence_length(text,words,sentences): 
    #words = word_count(text) 
    #sentences = sentence_count(text) 
    average_sentence_length = float(words / sentences) 
    return average_sentence_length 

#Returns Flesch Kincaid Score
def flesch_kincaid(text,avg_sen_len,avg_syl): 
    flesch = 206.835 - float(1.015 * avg_sen_len) -\
          float(84.6 * avg_syl) 
    return legacy_round(flesch, 2) 


##############################################FEATURES################################
#Features are being stored in feature set


#Number of Words (0)

for i in range(0,len(x_train)):
	text=x_train[i]
	wordcount=word_count(text)
	feature_set[i].append(wordcount)

#Number of Sentences (1)
for i in range(0,len(x_train)):
	text=x_train[i]
	sentencecount=sentence_count(text)
	feature_set[i].append(sentencecount)

#Number of Syllables (2)
for i in range(0,len(x_train)):
	text=x_train[i]
	syl_count=num_syllables(text)
	feature_set[i].append(syl_count)

#Average Number of Syllables per word (3)
for i in range(0,len(x_train)):
	text=x_train[i]
	avg_syl=avg_syllables_per_word(text)
	feature_set[i].append(avg_syl)

#Average Number of Words per sentence(4)
for i in range(0,len(x_train)):
	text=x_train[i]
	avg_syl=avg_sentence_length(text,feature_set[i][0],feature_set[i][1])
	feature_set[i].append(avg_syl)

#Flesch Kincaid Score (5)
for i in range(0,len(x_train)):
	text=x_train[i]
	flesch=flesch_kincaid(text,feature_set[i][4],feature_set[i][3])
	feature_set[i].append(flesch)

#Number of big words
for i in range(0,len(x_train)):
	text=x_train[i]
	sentences = break_sentences(text)
	big_word_count=0
	for item in sentences:
		for token in item:
			if(len(text)>=6)
			big_word_count=big_word_count+1;
	feature_set[i].append(big_word_count)

#Number of Short Sentences and Long Sentences (6,7)
for i in range(0,len(x_train)):
	text=x_train[i]
	sentences = break_sentences(text) 
	num_short=0
	num_long=0
	for item in sentences:
		if len(item)<=short_sentence_cutoff:
			num_short=num_short+1
		else:
			num_long=num_long+1
	feature_set[i].append(num_short)
	feature_set[i].append(num_long)

#Modal Verbs
list_modal_verbs=['can','could','may','might','must','shall','should','will','would']

for i in range(0,len(x_train)):
	text=x_train[i]
	sentences = break_sentences(text) 
	num_mod=0
	for item in sentences:
		for token in item:
			if token in list_modal_verbs:
				num_mod=num_mod+1
	feature_set[i].append(num_mod)

list_conjunctions=['and','but','for', 'nor', 'or', 'so', 'yet']


#######################################################################################

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

