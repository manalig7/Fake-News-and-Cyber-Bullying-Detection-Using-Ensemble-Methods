f=open('data.csv','r')
g=open('tweet_id','w')

for line in f:
	ls=line.split(',')
	g.write(ls[0]+'\n')