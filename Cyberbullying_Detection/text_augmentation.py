#https://towardsdatascience.com/data-augmentation-for-text-data-obtain-more-data-faster-525f7957acc9

tsv = 'Spell_Check_and_Segmentation/finalcb_clean_train_py_spell_check_segment_all.txt'

f=open(tsv,'r')
x_train=[]
y_train=[]
for line in f:
	ls=line.split('\t')
	x_train.append((ls[0]))
	y_train.append(int(ls[1]))


f.close()
"""
tsv = 'Spell_Check_and_Segmentation/finalcb_clean_test_py_spell_check_segment_all.txt'
f=open(tsv,'r')
x_test=[]
y_test=[]
for line in f:
	ls=line.split('\t')
	x_test.append((ls[0]))
	y_test.append(int(ls[1]))


f.close()
"""

from nltk import sent_tokenize
import json
import random
def tokenize(text):
    '''text: list of text documents'''
    tokenized =  sent_tokenize(text)
    return tokenized
	
		
def shuffle_tokenized(text):
    random.shuffle(text)
    newl=list(text)
    shuffled.append(newl)
    return text


augmented = []
reps=[]
"""
for ng_rev in data['negative']:
    tok = tokenize(ng_rev)
    shuffled= [tok]
    #print(ng_rev)
    for i in range(11):
	#generate 11 new reviews
        shuffle_tokenized(shuffled[-1])
    for k in shuffled:
		'''create new review by joining the shuffled sentences'''
        s = ' '
        new_rev = s.join(k)
        if new_rev not in augmented:
            augmented.append(new_rev)
        else:
            reps.append(new_rev)

"""
for i in range(0,len(x_train)):
	if y_train[i]==1:
		tok = tokenize(x_train[i])
		shuffled= [tok]	
		for i in range(0,2):
			shuffle_tokenized(shuffled[-1])
		for k in shuffled:
			s = ' '
			new_rev = s.join(k)
			if new_rev not in augmented:
				augmented.append(new_rev)
			else:
				reps.append(new_rev)

for item in augmented:
	x_train.append(item)
	y_train.append(1)


f=open('Spell_Check_and_Segmentation/augmented_train.tsv','w')

for i in range(0,len(x_train)):
	f.write(x_train[i]+'\t'+str(y_train[i])+'\n')







