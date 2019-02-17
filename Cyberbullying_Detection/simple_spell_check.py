#from espeakng import ESpeakNG


import re

import nltk
#nltk.download('words')



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


from spellchecker import SpellChecker

spell = SpellChecker()


f=open('Final_CB_DS/finalcb_dataset_cleaned.txt','r')
g=open('Final_CB_DS/spell_check.txt','w')

for line in f:
	ls=line.split('\t')
	text=ls[0]
	text=text.lower()
	val=int(ls[1])
	text=' '.join(text.split())
	text_words=text.split(' ')
	new_text=''
	for word in text_words:
		word=word.lower()
		if word.isalpha():
			converted_word=spell.correction(word)
			new_text=new_text+converted_word+' '
			continue
		elif word[:len(word)-1].isalpha():
			if word[len(word)-1]=='.':
				sliced_word=word[:len(word)-1]
				converted_word=spell.correction(word)
				new_text=new_text+converted_word+'.'+' '
				continue
			elif word[len(word)-1]==',':
				sliced_word=word[:len(word)-1]
				converted_word=spell.correction(word)
				new_text=new_text+converted_word+','+' '
				continue
			elif word[len(word)-1]=='!':
				sliced_word=word[:len(word)-1]
				converted_word=spell.correction(word)
				new_text=new_text+converted_word+'!'+' '
				continue
		new_text=new_text+word+' '
		
	g.write(new_text+'\t'+str(val)+'\n')





