#from espeakng import ESpeakNG
import phonetics
import sys
import re

"""
f=open('sample.txt','r')

for word in f:
	word=word.split('\n')
	word=word[0]
	#esng = ESpeakNG() 
	#ipa = esng.g2p ('Hello World!', ipa=2)
	phone=phonetics. soundex(word)
	print phone
"""

#PHONETIC REPRESENTATION OF THE DATASET
f=open('Final_CB_DS/finalcb_dataset_cleaned.txt','r')

"""
for line in f:
	ls=line.split('\t')
	text=ls[0]
	val=int(ls[1])
	print text
	regex = re.compile('[^a-zA-Z ]')
	text=regex.sub('', text)
	text=' '.join(text.split())
	print text
	#Now we have text with only spaces
	words=text.split(' ')
	result_text=''
	for word in words:
		#print word
		phone=phonetics.soundex(word)
		result_text=result_text+phone+' '
	result_text=result_text[:-1]
	print result_text
	sys.exit()

"""
s=set()
for line in f:
	ls=line.split('\t')
	text=ls[0]
	val=int(ls[1])
	regex = re.compile('[^a-zA-Z ]')
	sp_removed=regex.sub('', text)
	sp_removed=' '.join(sp_removed.split())
	word_list=sp_removed.split(' ')
	for i in range(0,len(word_list)):
		word_list[i]=word_list[i].lower()
	s=s.union(set(word_list))
	#print s
	#sys.exit()

print len(s)

vocab=list(s)

phenome=[]
conversion={}

for word in vocab:
	phenome.append(phonetics.soundex(word))






