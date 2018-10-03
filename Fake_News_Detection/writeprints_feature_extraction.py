"""
CHARACTER BASED FEATURES
Total characters-
Percentage of digits-
Percentage of letters-
Percentage of Uppercase letters-
Percentage of whitespace-
Percentage of tab spaces (NOT DOING)-
Frequency of each letter (26)-
Frequency of special characters (~ , @, #, $, %, ^, &, *, -, _, = ,+, >, <, [, ], {, }, /, \, |)-

WORD BASED FEATURES
Total number of words-
Percentage number of short words (less than 4 chracters)-
Percentage of characters in words-
Average word length-
Average sentence length in terms of character-
Average sentence length in terms of word-
Total different words-
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
Position of quoted content -Quoted content is below or above the replying body (Not doing)
Indentation of paragraph -Has indentation before each paragraph (Not doing)
Use e-mail as signature (Not doing)
Use telephone as signature (Not doing)
Use url as signature (Not doing)

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
import string

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

#Percentage number of short words (less than 4 chracters) and Percentage of characters in words, Average word length(32,33,34)
for i in range(0,len(x_train)):
	text=x_train[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(" ")
	count=0
	num_char=0
	for item in ls:
		if len(item)<4:
			count=count+1
		num_char=num_char+len(item)
	shortratio=float(count)/feature_set[i][31]
	word_char_ratio=float(num_char)/feature_set[i][0]
	av_word_length=float(num_char)/feature_set[i][31]
	feature_set[i].append(shortratio)
	feature_set[i].append(word_char_ratio)

#Average sentence length in terms of character (35) (Measuring all the characters including whitespaces)
for i in range(0,len(x_train)):
	text=x_train[i]
	count=0
	avg=0
	num_sentences=0
	for j in range(0,len(text)):
		if text[j]=='!' or text[j]=='?' or text[j]=='.':
			avg=avg+count
			num_sentences=num_sentences+1
			count=0
		else:
			count=count+1
	avg=float(avg)/num_sentences
	feature_set[i].append(avg)


#Average sentence length in terms of word (36) -Number of words by number of sentences
for i in range(0,len(x_train)):
	text=x_train[i]
	num_sentences=0
	for j in range(0,len(text)):
		if text[j]=='!' or text[j]=='?' or text[j]=='.':
			num_sentences=num_sentences+1
	avg_sen_len_words=float(feature_set[i][31])/num_sentences
	feature_set[i].append(avg_sen_len_words)

#Total different words (37)
for i in range(0,len(x_train)):
	text=x_train[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(' ')
	s=set(ls)
	feature_set[i].append(len(s))

#Frequency of once-occurring words (38)


#Frequency of twice-occurring words (39)


#A vocabulary richness measure defined by Yule (40)


# vocabulary richness measure defined by Simpson (41)


#A vocabulary richness measure defined by Sichele (42)


#A vocabulary richness measure defined by Brune (43)


#A vocabulary richness measure defined by Honore (44)


#Word length frequency distribution /Mnumber of words(20 features) (45-64) Frequency of words in different length 


#SYNTACTIC FEATURES

#Frequency of punctuations (65)
for i in range(0,len(x_train)):
	text=x_train[i]
	num_punc=0
	for j in range(0,len(text)):
		if text[j] in string.punctuation:
			num_punc=num_punc+1
	feature_set[i].append(num_punc)

#Frequency of function words (66)
func_words=['a','between','in','nor','some','upon','about','both','including','nothing','somebody','us','above','but','inside','of','someone','used','after','by','into','off','something','via','all','can','is','on','such','we','although','cos','it','once','than','what','am','do','its','one','that','whatever','among','down','latter','onto','the','when','an','each','less','opposite','their','where','and','either','like','or','them','whether','another','enough','little','our','these','which','any','every','lots','outside','they','while','anybody','everybody','many','over','this','who','anyone','everyone','me','own','those','whoever','anything','everything','more','past','though','whom','are','few','most','per','through','whose','around','following','much','plenty','till','will','as','for','must','plus','to','with','at','from','my','regarding','toward','within','be','have','near','same','towards','without','because','he','need','several','under','worth','before','her','neither','she','unless','would','behind','him','no','should','unlike','yes','below','i','nobody','since','until','you','beside','if','none','so','up','your']

#STRUCTURAL FEATURES

#Total number of lines (67).   USELESS FEATURE!!!!!!!!!!!!!!!!!!
for i in range(0,len(x_train)):
	text=x_train[i]
	num_lines=0
	for j in range(0,len(text)):
		if text[j]=='\n':
			num_lines=num_lines+1
	feature_set[i].append(num_lines)

#Total number of sentences (68)
for i in range(0,len(x_train)):
	text=x_train[i]
	num_sentences=0
	for j in range(0,len(text)):
		if text[j]=='!' or text[j]=='?' or text[j]=='.':
			num_sentences=num_sentences+1
	feature_set[i].append(num_sentences)

#Total number of paragraphs (69)  USELESS FEATURE!!!!!!!!!!!!!!!!!!
for i in range(0,len(x_train)):
	text=x_train[i]
	num_paras=0
	for j in range(0,len(text)):
		if text[j]=='\n' and (text[j-1]=='!' or text[j-1]=='?' or text[j-1]=='.'):
			num_paras=num_paras+1
	feature_set[i].append(num_paras)

#Number of sentences per paragraph (70) USELESS FEATURE!!!!!!!!!!!!!!!!!!

#Number of characters per paragraph (71) USELESS FEATURE!!!!!!!!!!!!!!!!!!

#Number of words per paragraph (72) USELESS FEATURE!!!!!!!!!!!!!!!!!!


#Has a greeting (73)
greetings=["hello","good afternoon","good evening","good morning"]
for i in range(0,len(x_train)):
	text=x_train[i].lower()
	flag=0
	for j in range(0,len(greetings)):
		if greetings[j] in text:
			flag=1
			break
	feature_set[i].append(flag)

#Has separators between paragraphs (74)


#Has quoted content  -Cite original message as part of replying message (Measuring the number of quotes)(75)
quotes=['"']
for i in range(0,len(x_train)):
	text=x_train[i]
	num_quotes=0
	for j in range(0,len(text)):
		if text[j] in quotes:
			num_quotes=num_quotes+1
	print num_quotes
	break
	feature_set[i].append(num_quotes)	

#Has URL (76)

