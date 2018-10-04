"""
Minimal Feature Set: 
Number of unique words
Complexity
Gunning-Fog-
Character count with whitespace-
Character count without whitespace-
Average syllables per word-
Sentence count-
Average sentence length-
Flesch Kincaid-

Ones mentioned in the referred paper
number of
different words, lexical density, Gunning-Fog readability index,
character count without whitespace, average syllables
per word, sentence count, average sentence length, and an
alternative readability measure.
"""
import pandas as pd  
import numpy as np
import sys
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from textatistic import Textatistic

tsv = 'FakeNewsNet_Dataset/fakenewsnet_train.txt'
f=open(tsv,'r')
x=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x.append((ls[0].decode('utf-8')).lower())
	y_train.append(int(ls[1]))
f.close()

m=len(x)

tsv1 = 'FakeNewsNet_Dataset/fakenewsnet_test.txt'
f=open(tsv1,'r')
y_test=[]


for line in f:
	ls=line.split('\t')
	x.append((ls[0].decode('utf-8')).lower())
	y_test.append(int(ls[1]))
f.close()

feature_set=[]

for i in range(0,len(x)):
	feature_set.append([])

x_train=feature_set[:m]
x_test=feature_set[m:]

#x contains the list of pieces of news texts
#1st feature- Quantity (number of syllables, number of words, number of sentences)

import spacy 
from textstat.textstat import easy_word_set, legacy_round 
from textstat.textstat import textstatistics
#from readcalc import readcalc
  
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
    return len(list(sentences)) 

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

   # Return total Difficult Words in a text 
def difficult_words(text): 
  	# Find all words in the text 
    words = [] 
    sentences = break_sentences(text) 
    for sentence in sentences: 
        words += [token for token in sentence] 
    diff_words_set = set() 
    for word in words: 
        syllable_count = num_syllables(str(word)) 
        if word not in easy_word_set and syllable_count >= 2: 
            diff_words_set.add(word) 
"""
def gunning_fog(text,wordcount,avg_sen_len): 
    per_diff_words = (difficult_words(text) / wordcount * 100) + 5
    grade = 0.4 * (avg_sen_len + per_diff_words) 
    return grade 
"""

##############################################FEATURES################################
#Features are being stored in feature set

"""#Number of Words 
wordcount=[]

for i in range(0,len(x)):
    text=x[i]
    wordcount.append(word_count(text))
    

#Number of Sentences (0)
for i in range(0,len(x)):
    text=x[i]
    sentencecount=sentence_count(text)
    feature_set[i].append(sentencecount)

#Average Number of Syllables per word (1)
for i in range(0,len(x)):
    text=x[i]
    avg_syl=avg_syllables_per_word(text)
    feature_set[i].append(avg_syl)

#Average Number of Words per sentence(2)
for i in range(0,len(x)):
    text=x[i]
    avg_sen_len=avg_sentence_length(text,wordcount[i],feature_set[i][0])
    feature_set[i].append(avg_sen_len)

#Flesch Kincaid Score (3)
for i in range(0,len(x)):
    text=x[i]
    flesch=flesch_kincaid(text,feature_set[i][2],feature_set[i][1])
    feature_set[i].append(flesch)

#Gunning-Fog (4)
for i in range(0,len(x)):
    text=x[i]
    #gf=gunning_fog(text,wordcount[i],feature_set[i][2])
    #gf=textstatistics.gunning_fog(self,text)
    #calc = readcalc.ReadCalc(text)
    #gf=calc.get_fog_index()
    print type(text)
    s = Textatistic(text)
    gf=a.gunningfog_score
    feature_set[i].append(gf)

#Number of characters with whitespace (5)
for i in range(0,len(x)):
    text=x[i]
    num_char_w=len(text)
    feature_set[i].append(num_char_w)
"""

wordcount=[]
#########################################################################
for i in range(0,len(x)):
	text=x[i]

#Number of Words 
    wordcount.append(word_count(text))
    
#Number of Sentences (0)
    sentencecount=sentence_count(text)
    feature_set[i].append(sentencecount)

#Average Number of Syllables per word (1)
    avg_syl=avg_syllables_per_word(text)
    feature_set[i].append(avg_syl)

#Average Number of Words per sentence(2)
    avg_sen_len=avg_sentence_length(text,wordcount[i],feature_set[i][0])
    feature_set[i].append(avg_sen_len)

#Flesch Kincaid Score (3)
    flesch=flesch_kincaid(text,feature_set[i][2],feature_set[i][1])
    feature_set[i].append(flesch)

#Gunning-Fog (4)
    s = Textatistic(text)
    gf=a.gunningfog_score
    feature_set[i].append(gf)

#Number of characters with whitespace (5)
    num_char_w=len(text)
    feature_set[i].append(num_char_w)

#Number of characters without whitespace (6)
	num_char=0
	for j in range(0,len(text)):
		if text[j]!=' ':
			num_char=num_char+1
	feature_set[i].append(num_char)

tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, news in enumerate(x):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(news)]
    tokenized_corpus.append(tokens)

#Number of unique words (7)
for i in range(0,len(tokenized_corpus)):
    text=tokenized_corpus[i]
    s=set(text)
    unq=len(s)
    feature_set[i].append(len(s))

#Lexical Density or Complexity- Number of Unique Tokens Divided by total number of words (8)
for i in range(0,len(x)):
    lex_density=feature_set[i][7]/float(wordcount[i])
    feature_set[i].append(lex_density)


#######################################################################################

print("################# Naive Bayes Classifier ####################")

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score

clf = MultinomialNB()
clf.fit(x_train,y_train)

print "\nAccuracy on Training Set :"
print clf.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(x_test, y_test)

y_pred=clf.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


print ("################### Random Forest Classifier ###############")

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier()
clf.fit(x_train,y_train)

print "\nAccuracy on Training Set :"
print clf.score(x_train, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(x_test, y_test)

y_pred=clf.predict(x_test)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


