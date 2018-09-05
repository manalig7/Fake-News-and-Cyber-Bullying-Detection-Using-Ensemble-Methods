
cap_in_fake=0
cap_in_real=0
total_fake=0
total_real=0
f=open('buzzfeed_news_content.txt')

for line in f:
	ls=line.split('\t')
	string=ls[0]
	val=int(ls[1])
	count=0
	total_count=0
	for char in string:
		if char.isupper():
			count=count+1
		total_count=total_count+1
	if val==1:
		cap_in_fake=cap_in_fake+count
		total_fake=total_fake+total_count
	else:
		cap_in_real=cap_in_real+count
		total_real=total_real+total_count

print "Number of Capital Letters in Fake News"
print cap_in_fake
print "Number of Capital Letters in Real News"
print cap_in_real

print "Ratio of Capital Letters in Fake News"
print float(cap_in_fake)/total_fake
print "Ratio of Capital Letters in Real News"
print float(cap_in_real)/total_real