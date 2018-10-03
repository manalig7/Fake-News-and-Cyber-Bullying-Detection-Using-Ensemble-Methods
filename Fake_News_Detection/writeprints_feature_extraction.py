"""
CHARACTER BASED FEATURES
Total characters
Percentage of digits
Percentage of letters
Percentage of Uppercase letters
Percentage of whitespace
Percentage of tab spaces (NOT DOING)
Frequency of each letter (26)
Frequency of special characters (~ , @, #, $, %, ^, &, *, -, _, = ,+, >, <, [, ], {, }, /, \, |)

WORD BASED FEATURES
Total number of words
Percentage number of short words (less than 4 chracters)
Percentage of characters in words
Average word length
Average sentence length in terms of character
Average sentence length in terms of word
Total different words
Frequency of once-occurring words
Frequency of twice-occurring words
A vocabulary richness measure defined by Yule
A vocabulary richness measure defined by Simpson
A vocabulary richness measure defined by Sichele
A vocabulary richness measure defined by Brune
A vocabulary richness measure defined by Honore
Word length frequency distribution /Mnumber of words(20 features) 
Frequency of words in different length

SYNTACTIC FEATURES
Frequency of punctuations
Frequency of function words

STRUCTURAL FEATURES
Total number of lines
Total number of sentences
Total number of paragraphs
Number of sentences per paragraph
Number of characters per paragraph
Number of words per paragraph
Has a greeting
Has separators between paragraphs
Has quoted content  -Cite original message as part of replying message
Position of quoted content -Quoted content is below or above the replying body
Indentation of paragraph -Has indentation before each paragraph
Use e-mail as signature
Use telephone as signature
Use url as signature

CONTENT SPECIFIC FEATURES
Frequency of content specific keyword
"""
import spacy 
from textstat.textstat import easy_word_set, legacy_round 
from textstat.textstat import textstatistics
from readcalc import readcalc
import pandas as pd  
import numpy as np
import sys
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

tsv = 'FakeNewsNet_Dataset/trial.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append((ls[0].decode('utf-8')))
	y_train.append(int(ls[1]))

feature_set=[]

for i in range(0,len(x_train)):
	feature_set.append([])

##############################FUNCTIONS#############################

def word_count(text): 
    sentences = break_sentences(text) 
    words = 0
    for sentence in sentences: 
        words += len([token for token in sentence]) 
    return words 

def break_sentences(text): 
    nlp = spacy.load('en') 
    doc = nlp(text) 
    return doc.sents

##############################FEATURES#############################

#CHARACTER BASED FEATURES
#Number of Characters (0)
for i in range(0,len(x_train)):
	text=x_train[i]
	charcount=len(text)
	feature_set[i].append(charcount)

#Percentage of Digits (1)
for i in range(0,len(x_train)):
	text=x_train[i]
	digcount=0
	for j in range(0,len(text)):
		if text[j].isdigit():
			digcount=digcount+1
	digratio=float(digcount)/feature_set[i][0]
	feature_set[i].append(digratio)

#Percentage of Upper case letters (2)
for i in range(0,len(x_train)):
	text=x_train[i]
	upcount=0
	for j in range(0,len(text)):
		if text[j].isupper():
			upcount=upcount+1
	upratio=float(upcount)/feature_set[i][0]
	feature_set[i].append(upratio)

#Percentage of whitespace (3)
for i in range(0,len(x_train)):
	text=x_train[i]
	whitecount=0
	for j in range(0,len(text)):
		if text[j]==" ":
			whitecount=whitecount+1
	whiteratio=float(whitecount)/feature_set[i][0]
	feature_set[i].append(whiteratio)

#Frequency of each letter (4-29)
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for i in range(0,len(x_train)):
	text=x_train[i]
	for j in range(0,len(letters)):
		count=0
		for k in range(0,len(text)):
			if text[k]==letters[j]:
				count=count+1
		feature_set[i].append(count)

#Frequency of special characters (30)
spchar=['~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|']
for i in range(0,len(x_train)):
	text=x_train[i]
	count=0
	for k in range(0,len(text)):
			if text[k] in spchar:
				count=count+1
	feature_set[i].append(count)

#WORD BASED FEATURES
char_remove=['.',',' ,'~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|','"','\'','!']

#Total number of words (31)
for i in range(0,len(x_train)):
	text=x_train[i]
	for item in char_remove:
		text=text.replace(item,"")
	wordcount=word_count(text)
	feature_set[i].append(wordcount)

#Percentage number of short words (less than 4 chracters) (32)
for i in range(0,len(x_train)):
	text=x_train[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(" ")
	count=0
	for item in ls:
		if len(item)<4:
			count=count+1
	shortratio=float(count)/feature_set[i][31]
	feature_set[i].append(shortratio)

#Percentage of characters in words (33)