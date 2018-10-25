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
Number of Conjunctions -
4. Uncertainity
Number of words expressing uncertainity
Number of tentative words -
Modal Verbs-
5. Specificity and Expressiveness
Rate of adjectives and adverbs  - 
Number of affective terms 
6. Verbal Non-Immediacy
Self References 
Number of first, second and third person pronoun usage -

"""


import pandas as pd  
import numpy as np
import sys
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

short_sentence_cutoff=10

"""tsv = 'FakeNewsNet_Dataset/trial1.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append(ls[0].lower())
	y_train.append(int(ls[1]))

feature_set=[]

for i in range(0,len(x_train)):
	feature_set.append([])"""

tsv = 'Finaldataset/finaldataset_train.txt'
f=open(tsv,'r')
x=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x.append((ls[0]).lower())
	y_train.append(int(ls[1]))
f.close()

m=len(x)

tsv1 = 'Finaldataset/finaldataset_test.txt'
f=open(tsv1,'r')
y_test=[]


for line in f:
	ls=line.split('\t')
	x.append((ls[0]).lower())
	y_test.append(int(ls[1]))
f.close()

feature_set=[]

for i in range(0,len(x)):
	feature_set.append([])

x_train=feature_set[:m]
x_test=feature_set[m:]

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
    words = 0.00001
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
    average_sentence_length = float(words / (sentences+0.00001)) 
    return average_sentence_length 

#Returns Flesch Kincaid Score
def flesch_kincaid(text,avg_sen_len,avg_syl): 
    flesch = 206.835 - float(1.015 * avg_sen_len) -\
          float(84.6 * avg_syl) 
    return legacy_round(flesch, 2) 


##############################################FEATURES################################
#Features are being stored in feature set


#Number of Words (0)

for i in range(0,len(x)):
	text=x[i]
	wordcount=word_count(text)
	feature_set[i].append(wordcount)
	
	#Number of Sentences (1)
	sentencecount=sentence_count(text)
	feature_set[i].append(sentencecount)
	
	#Number of Syllables (2)
	syl_count=num_syllables(text)
	feature_set[i].append(syl_count)
	
	#Average Number of Syllables per word (3)
	avg_syl=avg_syllables_per_word(text)
	feature_set[i].append(avg_syl)
	
	#Average Number of Words per sentence(4)
	avg_syl=avg_sentence_length(text,feature_set[i][0],feature_set[i][1])
	feature_set[i].append(avg_syl)
	
	#Flesch Kincaid Score (5)
	flesch=flesch_kincaid(text,feature_set[i][4],feature_set[i][3])
	feature_set[i].append(flesch)
	
	sentences = break_sentences(text)
	big_word_count=0
	num_short=0
	num_long=0
	for item in list(sentences):
		if len(item)<=short_sentence_cutoff:
			num_short=num_short+1
		else:
			num_long=num_long+1
		for token in item:
			if len(token)>=6:
				big_word_count=big_word_count+1;
	feature_set[i].append(big_word_count)
	
	feature_set[i].append(num_short)
	
	feature_set[i].append(num_long)
	


char_remove=['.',',' ,'~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|','"','\'','!']

#Modal Verbs
list_modal_verbs=['can','could','may','might','must','shall','should','will','would']
list_conjunctions=['and','but','for', 'nor', 'or', 'so', 'yet', 'because', 'if' ,'as', 'since']
list_tentative_words=[ 'suggests that' , 'appear' ,'indicate', 'may', 'might','could','can','possibly', 'probably','likely','perhaps','uncertain','maybe',' conceivably' ,'tentative','tentatively' ,'tends to' , 'seems to']
list_first_person_pronouns=['I','we','me','us','my','our','mine','ours']
list_second_person_pronouns=['you','your','yours']
list_third_person_pronouns=['he','they','him','them','his', 'her','their','she','her','hers','theirs','it','its']

for j in range(0,len(feature_set)):
	print (len(feature_set[j]),end=',')

for i in range(0,len(x)):
	text=x[i].lower()
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(" ")
	count_modal=0
	num_conjunct=0
	num_tentative=0
	num_first_person=0
	num_second_person=0
	num_third_person=0
	num_adj_adv=0;


	"""para = word_tokenize(text)
				tags = nltk.pos_tag(para)
			
				counts = Counter(tag for word,tag in tags)
				
				num_adj_adv = counts['JJ'] + counts['JJR'] +counts['JJS'] + counts['RB'] + counts['RBR'] +counts['RBS']"""
	#print ("number of adj adv",type(num_adj_adv))
	words = word_count(text) 

	rate_adj_adv = num_adj_adv/words
	#print ("rate_adj_adv : ",rate_adj_adv)
	for item in ls:
		if item in list_modal_verbs:
			count_modal=count_modal+1
		if item in list_conjunctions:
			num_conjunct=num_conjunct+1
		if item in list_tentative_words:
			num_tentative=num_tentative+1
		if item in list_first_person_pronouns:
			num_first_person=num_first_person+1
		elif item in list_second_person_pronouns:
			num_second_person=num_second_person+1
		elif item in list_third_person_pronouns:
			num_third_person=num_third_person+1

	feature_set[i].append(count_modal)
		
	feature_set[i].append(num_conjunct)
		
	feature_set[i].append(num_tentative)
		
	feature_set[i].append(num_first_person)
		
	feature_set[i].append(num_second_person)
		
	feature_set[i].append(num_third_person)
		
		#feature_set[i].append(rate_adj_adv)
		

print ("################### Random Forest Classifier ###############")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score

clf = RandomForestClassifier()

clf.fit(x_train,y_train)

print ("\nAccuracy on Training Set :")
print (clf.score(x_train, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf.score(x_test, y_test))

y_pred=clf.predict(x_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))