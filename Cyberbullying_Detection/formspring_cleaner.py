
import re

f=open('formspring_modified_1.tsv','r')
g=open('formspring_modified_2.tsv','w')
for line in f:
	ls=line.split('\t')
	text=ls[0]
	val=ls[1]
	text = re.sub('[@#$]0123456789%*&();:', '', text)
	g.write(text+'\t'+val)
