
from statistics import mode
from statistics import StatisticsError

f=open('formspring_data.csv','r')

dataset=[]
flag=0
for line in f:
	if flag==0:
		flag=1
		continue
	ls=line.split('\t')
	temp=[]
	try:
		stri=ls[2]+' '+ls[3]
	except IndexError:
		continue
	temp.append(stri)
	ans=[]
	try:
		ans.append(ls[5])
		ans.append(ls[8])
		ans.append(ls[11])
	except IndexError:
		continue
	try:
		final_ans=mode(ans)
	except StatisticsError:
		continue
	if final_ans=='No':
		num=0
	if final_ans=='Yes':
		num=1
	temp.append(num)
	dataset.append(temp)

f.close()

f=open('formspring_modified.tsv','w')

for i in range(0,len(dataset)):
	f.write(dataset[i][0]+'\t'+str(dataset[i][1])+'\n')