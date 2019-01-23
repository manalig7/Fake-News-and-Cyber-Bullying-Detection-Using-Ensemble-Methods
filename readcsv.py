import pandas as pd
import re

df = pd.read_csv("attack_annotated_comments.tsv",delimiter='\t')

[x,y]=df.shape
#print (df["rev_id"])


df1 = pd.read_csv("attack_annotations.tsv",delimiter='\t')
[x1,y1]=df1.shape
#print y1#
#print df1.iloc[:,0]

k=0
bully=0
nbully=0
elp1=0
elp2=0
for j in range(0,x) :
	attack=0
	#print k
	for i in range(k,x1):
		if df.iloc[j,0]==df1.iloc[i,0]:
			if (df1.iloc[i,6]==1):
				attack=attack+1	
			
		else:
			if attack>=1:
				bully=bully+1
				if bully<750:
					item=re.sub("NEWLINE_TOKEN","",df.iloc[j,1])
					print item+"\t"+str(1)
				else:
					elp1=1
			else:
				nbully=nbully+1
				if nbully<750:
					item=re.sub("NEWLINE_TOKEN","",df.iloc[j,1])
					print item+"\t"+str(0)
				else:
					elp2=1
			k=i
			break
	if elp1 and elp2:
		break


