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
import nltk
from nltk.tag import StanfordPOSTagger
from collections import Counter
from nltk.internals import find_jars_within_path
from textblob import TextBlob
import pickle
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

analyser = SentimentIntensityAnalyzer()
tokenizer = RegexpTokenizer(' ', gaps=True)
"""
tsv = 'Spell_Check_and_Segmentation/finalcb_clean_train_py_spell_check_segment_all_trial.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]
lent=[]
data=[]
for line in f:
	ls=line.split('\t')
	x_train.append((ls[0]))

	
	temp = []
	for j in tokenizer.tokenize(ls[0]):
		#print(j) 
	       	temp.append(j) 
	data.append(temp)

	lent.append(len(temp))

	y_train.append(int(ls[1]))

feature_set=[]

m=len(x_train)
for i in range(0,len(x_train)):
	feature_set.append([])

f.close()
"""

tsv = 'Spell_Check_and_Segmentation/finalcb_clean_test_py_spell_check_segment_all.txt'
f=open(tsv,'r')
x_test=[]
y_test=[]
datatest=[]
lentest=[]
for line in f:
	ls=line.split('\t')
	x_test.append((ls[0]))

	temp = []
	for j in tokenizer.tokenize(ls[0]):
		#print(j) 
	       	temp.append(j) 
	datatest.append(temp)

	lentest.append(len(temp))

	y_test.append(int(ls[1]))

feature_set_test=[]

for i in range(0,len(x_test)):
	feature_set_test.append([])

f.close()


medium_words=[]
strong_words=[]
mild_words=[]

f=open('ofcom_profane_strong.txt','r')
for line in f:
	ls=line.split()
	ls[0]=ls[0].lower()

	strong_words.append(ls[0])
f.close()

f=open('ofcom_profane_medium.txt','r')
for line in f:
	ls=line.split()
	ls[0]=ls[0].lower()

	medium_words.append(ls[0])
f.close()

f=open('ofcom_profane_mild.txt','r')
for line in f:
	ls=line.split()
	ls[0]=ls[0].lower()

	mild_words.append(ls[0])
f.close()

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

def hapaxlegomena(text):																	  
    freq = nltk.FreqDist(word for word in text.split())
    hapax_legomena = [key for key,val in freq.items() if val==1]
    return len(hapax_legomena)

def dislegomena(text):																	  
    freq = nltk.FreqDist(word for word in text.split())
    dislegomena = [key for key,val in freq.items() if val==2]
    return len(dislegomena)

def fit_transform(d,feature_set):
	res=[]
	for i in range(0,len(d)):
		temp=[]	
		for j in range(0,len(voc)):
			#print(voc[j])
			if voc[j] in d[i]:
				#print 	(np.mean(model_W2V.wv[voc[j]]))	
				#temp.append(np.mean(model_W2V.wv[voc[j]]))
				if voc[j] in mild_words:
					temp.append(2*feature_set[i][24][1])
				elif voc[j] in medium_words:
					temp.append(5*feature_set[i][24][1])
				elif voc[j] in strong_words:
					temp.append(8*feature_set[i][24][1])
				else:
					temp.append(-1*feature_set[i][24][2])
			else :
				temp.append(0)		
		res.append(temp)
	return res


from nltk.stem.porter import PorterStemmer
from itertools import groupby
 
def words(entry):
    return filter(lambda w: len(w), [w.strip("0123456789!:,.?(){}[]") for w in entry.split()])
 
def yules_K_calc(entry):
    freq_word = {}
    stemmer = PorterStemmer() #Stemming or reducing it to root form
    for w in words(entry):
        w = stemmer.stem(w).lower()
        try:
            freq_word[w] += 1 #Increase the frequency of the word
        except KeyError:
            freq_word[w] = 1 #Add new entry
 
    m1 = float(len(freq_word))
    m2 = sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(freq_word.values()))])
    try:
        yules_K= (m2-m1)/(m1*m1)
    except ZeroDivisionError:
        yules_K= 0
    return yules_K


pad_len=max(lent)

model_W2V = Word2Vec(data, size=10, window=5, min_count=1, workers=5, sg=0,max_vocab_size=10000)
model_W2V.save("CBOW_W2V.model")
model_W2V = Word2Vec.load("CBOW_W2V.model")

print("CBOW W2V model_done!")

voc=list(model_W2V.wv.vocab)
 

##############################FEATURES#############################

#CHARACTER BASED FEATURES
"""


#Frequency of each letter (4-29)
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


#Frequency of special characters (30)
spchar=['~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|']


#WORD BASED FEATURES
char_remove=['.',',' ,'~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|','"','\'','!']

#Total number of words (0)
for i in range(0,len(x_train)):
	text=x_train[i]
	for item in char_remove:
		text=text.replace(item,"")
	wordcount=word_count(text)
	feature_set[i].append(wordcount)

#Number of Characters (1)
for i in range(0,len(x_train)):
	text=x_train[i]
	charcount=len(text)
	feature_set[i].append(charcount)



#Total different words (2)
for i in range(0,len(x_train)):
	text=x_train[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(' ')
	s=set(ls)
	feature_set[i].append(len(s))


#Percentage number of short words (less than 4 chracters) and Percentage of characters in words, Average word length(3,4)
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
	if feature_set[i][0]!=0:
		shortratio=float(count)/feature_set[i][0]
	else:
		shortratio=0.0001
	if feature_set[i][1]!=0:
		word_char_ratio=float(num_char)/feature_set[i][1]
	else:
		word_char_ratio=0.0001
	if feature_set[i][0]!=0:
		av_word_length=float(num_char)/feature_set[i][0]
	else:
		av_word_length=0.0001
	feature_set[i].append(shortratio)
	feature_set[i].append(word_char_ratio)

#Frequency of function words ()
func_words=['a','between','in','nor','some','upon','about','both','including','nothing','somebody','us','above','but','inside','of','someone','used','after','by','into','off','something','via','all','can','is','on','such','we','although','cos','it','once','than','what','am','do','its','one','that','whatever','among','down','latter','onto','the','when','an','each','less','opposite','their','where','and','either','like','or','them','whether','another','enough','little','our','these','which','any','every','lots','outside','they','while','anybody','everybody','many','over','this','who','anyone','everyone','me','own','those','whoever','anything','everything','more','past','though','whom','are','few','most','per','through','whose','around','following','much','plenty','till','will','as','for','must','plus','to','with','at','from','my','regarding','toward','within','be','have','near','same','towards','without','because','he','need','several','under','worth','before','her','neither','she','unless','would','behind','him','no','should','unlike','yes','below','i','nobody','since','until','you','beside','if','none','so','up','your']

list_first_person_pronouns=['I','we','me','us','my','our','mine','ours']
list_second_person_pronouns=['you','your','yours']
list_third_person_pronouns=['he','they','him','them','his', 'her','their','she','her','hers','theirs','it','its']



	
#STRUCTURAL FEATURES

#Has a greeting (5)
greetings=["hello","good afternoon","good evening","good morning"]
for i in range(0,len(x_train)):
	text=x_train[i].lower()
	flag=0
	for j in range(0,len(greetings)):
		if greetings[j] in text:
			flag=1
	feature_set[i].append(flag)


#Has quoted content  -Cite original message as part of replying message (Measuring the number of quotes)(6)
quotes=['"']
for i in range(0,len(x_train)):
	text=x_train[i]
	num_quotes=0
	for j in range(0,len(text)):
		if text[j] in quotes:
			num_quotes=num_quotes+1
	feature_set[i].append(num_quotes)	



char_remove=['.',',' ,'~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|','"','\'','!']



#A vocabulary richness measure defined by Yule (7)

for i in range(0,len(x_train)):
	text=x_train[i].lower()
	feature_set[i].append(yules_K_calc(text))


#POSTagger(8-20)
for i in range(0,len(x_train)):
	text=x_train[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(" ")
	
	para = word_tokenize(text)
	tags = nltk.pos_tag(para)
	counts = Counter(tag for word,tag in tags)
	
	jar = '/home/manali/Documents/Computer Science 7th Sem/Major Project/Major-Project/Cyberbullying_Detection/stanford-postagger-full-2018-10-16/stanford-postagger.jar'
	model = '/home/manali/Documents/Computer Science 7th Sem/Major Project/Major-Project/Cyberbullying_Detection/stanford-postagger-full-2018-10-16/models/english-left3words-distsim.tagger'

	pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

	text = pos_tagger.tag(word_tokenize(text))

	counts = Counter(tag for word,tag in text)

	feature_set[i].append(counts['NN'])	#singular or mass common noun
	feature_set[i].append(counts['NNS'])	#plural common noun
	feature_set[i].append(counts['NNP'])	#singular proper noun
	feature_set[i].append(counts['JJ'])	#adjective
	feature_set[i].append(counts['JJR'])	#comparative adjective
	feature_set[i].append(counts['JJS'])	#superlative adjective
	feature_set[i].append(counts['MD'])	#modal auxillary
	feature_set[i].append(counts['RB'])	#adverb 
	feature_set[i].append(counts['RBR'])	#comparative adverb
	feature_set[i].append(counts['RBS'])	#superlative adverb
	feature_set[i].append(counts['UH'])	#interjection
	feature_set[i].append(counts['VB']+counts['VBD']+counts['VBG']+counts['VBN']+counts['VBP']+counts['VBZ'])	#verb base form, past tense, present participle or gerund, past participle, present tense not 3rd person, present tense 3rd person, 
	feature_set[i].append(counts['WDT']+counts['WP']+counts['WRB'])  #WH - determiner, pronoun, adverb

#First,second,third person pronouns(21,22,23)
for i in range(0,len(x_train)):
	text=x_train[i].lower()
	for item in char_remove:
		text=text.replace(item,"")
	num_first_person=0
	num_second_person=0
	num_third_person=0
	ls=text.split(' ')
	for j in range(0,len(ls)):
		if ls[j] in list_first_person_pronouns:
			num_first_person=num_first_person+1
		elif ls[j] in list_second_person_pronouns:
			num_second_person=num_second_person+1
		elif ls[j] in list_third_person_pronouns:
			num_third_person=num_third_person+1


	feature_set[i].append(num_first_person)
	feature_set[i].append(num_second_person)
	feature_set[i].append(num_third_person)


#sentiment polarity score(24)
#LIST OF PROFANIC WORDS- MILD, MEDIUM AND STRONG(25,26,27)
for i in range(0,len(x_train)):

	reslist=[]
	ls=x_train[i].split('\t')
	# testimonial = TextBlob(ls[0])
	# res= testimonial.sentiment
	# reslist.extend([res[0],res[1]])
	# #print reslist
	score = analyser.polarity_scores(ls[0])
	## merge this reslist to feature_vector in Writeprints
	print(score)
	reslist.extend(score.values())
	#print "P="+str(res[0])
	#print "S="+str(res[1])
	feature_set[i].append(reslist)
	print("sentiment:",reslist)


	profane_vec=[]
	text = x_train[i].lower()
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(' ')

	mild_profane=0
	medium_profane=0
	strong_profane=0
	for item in ls:
		if item in mild_words:
			mild_profane=mild_profane+1
		elif item in medium_words:
			medium_profane=medium_profane+1
		elif item in strong_words:
			strong_profane=strong_profane+1

	#print(len(voc))

	feature_set[i].append(mild_profane)
	feature_set[i].append(medium_profane)
	feature_set[i].append(strong_profane)

#for i in range(0,len(feature_set)):
#	print(feature_set[i])


XVAL = fit_transform(data,feature_set)

# for i in range(0,len(XVAL)):
# 	print(XVAL[i])

x_train=XVAL[:]

"""

##############################FEATURES FOR TEST###############################

#CHARACTER BASED FEATURES



#Frequency of each letter (4-29)
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


#Frequency of special characters (30)
spchar=['~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|']


#WORD BASED FEATURES
char_remove=['.',',' ,'~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|','"','\'','!']

#Total number of words (0)
for i in range(0,len(x_train)):
	text=x_test[i]
	for item in char_remove:
		text=text.replace(item,"")
	wordcount=word_count(text)
	feature_set_test[i].append(wordcount)

#Number of Characters (1)
for i in range(0,len(x_test)):
	text=x_test[i]
	charcount=len(text)
	feature_set_test[i].append(charcount)



#Total different words (2)
for i in range(0,len(x_test)):
	text=x_test[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(' ')
	s=set(ls)
	feature_set_test[i].append(len(s))


#Percentage number of short words (less than 4 chracters) and Percentage of characters in words, Average word length(3,4)
for i in range(0,len(x_test)):
	text=x_test[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(" ")
	count=0
	num_char=0
	for item in ls:
		if len(item)<4:
			count=count+1
		num_char=num_char+len(item)
	if feature_set_test[i][0]!=0:
		shortratio=float(count)/feature_set_test[i][0]
	else:
		shortratio=0.0001
	if feature_set_test[i][1]!=0:
		word_char_ratio=float(num_char)/feature_set_test[i][1]
	else:
		word_char_ratio=0.0001
	if feature_set_test[i][0]!=0:
		av_word_length=float(num_char)/feature_set_test[i][0]
	else:
		av_word_length=0.0001
	feature_set_test[i].append(shortratio)
	feature_set_test[i].append(word_char_ratio)

#Frequency of function words ()
func_words=['a','between','in','nor','some','upon','about','both','including','nothing','somebody','us','above','but','inside','of','someone','used','after','by','into','off','something','via','all','can','is','on','such','we','although','cos','it','once','than','what','am','do','its','one','that','whatever','among','down','latter','onto','the','when','an','each','less','opposite','their','where','and','either','like','or','them','whether','another','enough','little','our','these','which','any','every','lots','outside','they','while','anybody','everybody','many','over','this','who','anyone','everyone','me','own','those','whoever','anything','everything','more','past','though','whom','are','few','most','per','through','whose','around','following','much','plenty','till','will','as','for','must','plus','to','with','at','from','my','regarding','toward','within','be','have','near','same','towards','without','because','he','need','several','under','worth','before','her','neither','she','unless','would','behind','him','no','should','unlike','yes','below','i','nobody','since','until','you','beside','if','none','so','up','your']

list_first_person_pronouns=['I','we','me','us','my','our','mine','ours']
list_second_person_pronouns=['you','your','yours']
list_third_person_pronouns=['he','they','him','them','his', 'her','their','she','her','hers','theirs','it','its']



	
#STRUCTURAL FEATURES

#Has a greeting (5)
greetings=["hello","good afternoon","good evening","good morning"]
for i in range(0,len(x_test)):
	text=x_test[i].lower()
	flag=0
	for j in range(0,len(greetings)):
		if greetings[j] in text:
			flag=1
	feature_set_test[i].append(flag)


#Has quoted content  -Cite original message as part of replying message (Measuring the number of quotes)(6)
quotes=['"']
for i in range(0,len(x_test)):
	text=x_test[i]
	num_quotes=0
	for j in range(0,len(text)):
		if text[j] in quotes:
			num_quotes=num_quotes+1
	feature_set_test[i].append(num_quotes)	



char_remove=['.',',' ,'~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|','"','\'','!']



#A vocabulary richness measure defined by Yule (7)

for i in range(0,len(x_test)):
	text=x_test[i].lower()
	feature_set_test[i].append(yules_K_calc(text))


#POSTagger(8-20)
for i in range(0,len(x_test)):
	text=x_test[i]
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(" ")
	
	para = word_tokenize(text)
	tags = nltk.pos_tag(para)
	counts = Counter(tag for word,tag in tags)
	
	jar = '/home/manali/Documents/Computer Science 7th Sem/Major Project/Major-Project/Cyberbullying_Detection/stanford-postagger-full-2018-10-16/stanford-postagger.jar'
	model = '/home/manali/Documents/Computer Science 7th Sem/Major Project/Major-Project/Cyberbullying_Detection/stanford-postagger-full-2018-10-16/models/english-left3words-distsim.tagger'

	pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

	text = pos_tagger.tag(word_tokenize(text))

	counts = Counter(tag for word,tag in text)

	feature_set_test[i].append(counts['NN'])	#singular or mass common noun
	feature_set_test[i].append(counts['NNS'])	#plural common noun
	feature_set_test[i].append(counts['NNP'])	#singular proper noun
	feature_set_test[i].append(counts['JJ'])	#adjective
	feature_set_test[i].append(counts['JJR'])	#comparative adjective
	feature_set_test[i].append(counts['JJS'])	#superlative adjective
	feature_set_test[i].append(counts['MD'])	#modal auxillary
	feature_set_test[i].append(counts['RB'])	#adverb 
	feature_set_test[i].append(counts['RBR'])	#comparative adverb
	feature_set_test[i].append(counts['RBS'])	#superlative adverb
	feature_set_test[i].append(counts['UH'])	#interjection
	feature_set_test[i].append(counts['VB']+counts['VBD']+counts['VBG']+counts['VBN']+counts['VBP']+counts['VBZ'])	#verb base form, past tense, present participle or gerund, past participle, present tense not 3rd person, present tense 3rd person, 
	feature_set_test[i].append(counts['WDT']+counts['WP']+counts['WRB'])  #WH - determiner, pronoun, adverb

#First,second,third person pronouns(21,22,23)
for i in range(0,len(x_test)):
	text=x_test[i].lower()
	for item in char_remove:
		text=text.replace(item,"")
	num_first_person=0
	num_second_person=0
	num_third_person=0
	ls=text.split(' ')
	for j in range(0,len(ls)):
		if ls[j] in list_first_person_pronouns:
			num_first_person=num_first_person+1
		elif ls[j] in list_second_person_pronouns:
			num_second_person=num_second_person+1
		elif ls[j] in list_third_person_pronouns:
			num_third_person=num_third_person+1


	feature_set_test[i].append(num_first_person)
	feature_set_test[i].append(num_second_person)
	feature_set_test[i].append(num_third_person)


#sentiment polarity score(24)
#LIST OF PROFANIC WORDS- MILD, MEDIUM AND STRONG(25,26,27)
for i in range(0,len(x_test)):

	reslist=[]
	ls=x_test[i].split('\t')
	# testimonial = TextBlob(ls[0])
	# res= testimonial.sentiment
	# reslist.extend([res[0],res[1]])
	# #print reslist
	score = analyser.polarity_scores(ls[0])
	## merge this reslist to feature_vector in Writeprints
	print(score)
	reslist.extend(score.values())
	#print "P="+str(res[0])
	#print "S="+str(res[1])
	feature_set_test[i].append(reslist)
	print("sentiment:",reslist)


	profane_vec=[]
	text = x_test[i].lower()
	for item in char_remove:
		text=text.replace(item,"")
	ls=text.split(' ')

	mild_profane=0
	medium_profane=0
	strong_profane=0
	for item in ls:
		if item in mild_words:
			mild_profane=mild_profane+1
		elif item in medium_words:
			medium_profane=medium_profane+1
		elif item in strong_words:
			strong_profane=strong_profane+1

	#print(len(voc))

	feature_set_test[i].append(mild_profane)
	feature_set_test[i].append(medium_profane)
	feature_set_test[i].append(strong_profane)

#for i in range(0,len(feature_set)):
#	print(feature_set[i])


XVAL_test = fit_transform(datatest,feature_set_test)

x_test=XVAL_test[:]

# for i in range(0,len(XVAL)):
# 	print(XVAL[i])

"""

f=open('finalcb_senti_profane_weighted_features_train.tsv','w')
for i in range(0,len(x_train)):
	for j in range(0,len(x_train[i])):
		f.write(str(x_train[i][j])+"\t")
	f.write(str(y_train[i]))
	f.write("\n")
f.close()


f=open('finalcb_senti_profane_weighted_features_test.tsv','w')
for i in range(0,len(x_test)):
	for j in range(0,len(x_test[i])):
		f.write(str(x_test[i][j])+"\t")
	f.write(str(y_test[i]))
	f.write("\n")
f.close()
"""
"""

print("################# Naive Bayes Classifier ####################")

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
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


clf = RandomForestClassifier(random_state=20)
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


print ("################### Logistic regression Classifier ###############")

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=20, solver='lbfgs')

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
"""
"""
print ("################### SVM Classifier ###############")

from sklearn.svm import SVC

clf = SVC(gamma='auto')
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
"""