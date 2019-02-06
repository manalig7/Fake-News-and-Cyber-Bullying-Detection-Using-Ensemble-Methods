#from espeakng import ESpeakNG
import phonetics
import sys
import re
import epitran
import nltk
#nltk.download('words')
from nltk.corpus import words
word_corpus_nltk = words.words()
for i in range(0,len(word_corpus_nltk)):
	word_corpus_nltk[i]=word_corpus_nltk[i].lower()

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
import epitran.flite
fl = epitran.flite.Flite()

s=set()
for line in f:
	ls=line.split('\t')
	text=ls[0]
	text=text.lower()
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

f.close()

print len(s)

vocab=list(s)

phenomes=[]
words=[]
conversion={}

vocab.remove('')

for word in vocab:
	#print word
	"""
	new_word=''
	if len(word)>2:
		for j in range(0,len(word)-2):
			if word[j]==word[j+1] and word[j]==word[j+2]:
				continue
			else: 
				new_word=new_word+word[j]
	word=new_word
	"""
	word=word.lower()
	p=fl.english_g2p(word)
	if p not in phenomes:
		words.append(word)
		phenomes.append(p)
		conversion[word.lower()]=word.lower()
	else:
		i=phenomes.index(p)
		if words[i].lower() not in word_corpus_nltk:
			temp=words[i]
			words[i]=word
			conversion[word.lower()]=word.lower()
			conversion[temp.lower()]=word.lower()
		else:
			conversion[word.lower()]=words[i].lower()


f=open('Final_CB_DS/finalcb_dataset_cleaned.txt','r')
g=open('Final_CB_DS/sample_phonetic_ipa_vocab.txt','w')

for line in f:
	ls=line.split('\t')
	text=ls[0]
	text=text.lower()
	val=int(ls[1])
	text=' '.join(text.split())
	text_words=text.split(' ')
	new_text=''
	for word in text_words:
		if word.isalpha():
			converted_word=conversion[word.lower()]
			new_text=new_text+converted_word+' '
			continue
		#new_text=new_text+word+' '
		"""
		new_word=''
		if len(word)>2:
			for j in range(0,len(word)-2):
				if word[j]==word[j+1] and word[j]==word[j+2]:
					continue
				else: 
					new_word=new_word+word[j]
		word=new_word
		"""
	
		if word.isalpha():
			converted_word=conversion[word.lower()]
			new_text=new_text+converted_word+' '
			continue
		elif word[:len(word)-1].isalpha():
			if word[len(word)-1]=='.':
				sliced_word=word[:len(word)-1]
				converted_word=conversion[sliced_word.lower()]
				new_text=new_text+converted_word+'.'+' '
				continue
			elif word[len(word)-1]==',':
				sliced_word=word[:len(word)-1]
				converted_word=conversion[sliced_word.lower()]
				new_text=new_text+converted_word+','+' '
				continue
			elif word[len(word)-1]=='!':
				sliced_word=word[:len(word)-1]
				converted_word=conversion[sliced_word.lower()]
				new_text=new_text+converted_word+'!'+' '
				continue
		new_text=new_text+word+' '
		
	g.write(new_text+'\t'+str(val)+'\n')



print "are and arrow"
print fl.english_g2p('herman')
print fl.english_g2p('hermann')
print fl.english_g2p('you')
print fl.english_g2p('yahoo')



