import sys
medium_words=[]
strong_words=[]
mild_words=[]


f=open('ofcom_profane_strong.txt','r')

for line in f:
	ls=line.split()
	strong_words.append(ls[0])

f.close()


f=open('ofcom_profane_medium.txt','r')

for line in f:
	ls=line.split()
	medium_words.append(ls[0])

f.close()


f=open('ofcom_profane_mild.txt','r')

for line in f:
	ls=line.split()
	mild_words.append(ls[0])

f.close()
