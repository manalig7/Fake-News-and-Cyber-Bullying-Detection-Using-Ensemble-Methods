
f=open('formspring_modified_1.tsv','r')

text=[]
labels=[]
count=0
maxi=776

for line in f:
	ls=line.split('\t')
	if int(ls[1])==1:
		text.append(ls[0])
		labels.append(int(ls[1]))
	else:
		if count<maxi:
			text.append(ls[0])
			labels.append(int(ls[1]))
			count=count+1

f.close()

g=open('selected_1500_with_sp_ch.txt','w')

for i in range(0,len(labels)):
	g.write(text[i]+'\t'+str(labels[i])+'\n')

g.close()