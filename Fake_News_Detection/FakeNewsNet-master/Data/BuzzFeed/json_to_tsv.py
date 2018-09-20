import os
import json
import re

for item in os.listdir('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/'):
	#f=open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/Buzzfeed/FakeNewsContent/'+str(item))
	print item
	data = json.loads(open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/BuzzFeed/FakeNewsContent/'+str(item)).read())
	#print data
	text=data['text']
	text=re.sub(r'[ \t\n\r]+', ' ', text)
	fake=1
	write_file=open('final_news_content.txt','a+')
	write_file.write(text.encode('utf-8').strip()+'\t'+str(fake)+'\n')

for item in os.listdir('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/'):
	#f=open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/Buzzfeed/FakeNewsContent/'+str(item))
	print item
	data = json.loads(open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/BuzzFeed/RealNewsContent/'+str(item)).read())
	#print data
	text=data['text']
	text=re.sub(r'[ \t\n\r]+', ' ', text)
	fake=0
	write_file=open('final_news_content.txt','a+')
	write_file.write(text.encode('utf-8').strip()+'\t'+str(fake)+'\n')


for item in os.listdir('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/'):
	#f=open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/Buzzfeed/FakeNewsContent/'+str(item))
	print item
	data = json.loads(open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/'+str(item)).read())
	#print data
	text=data['text']
	text=re.sub(r'[ \t\n\r]+', ' ', text)
	fake=1
	write_file=open('final_news_content.txt','a+')
	write_file.write(text.encode('utf-8').strip()+'\t'+str(fake)+'\n')

for item in os.listdir('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/PolitiFact/RealNewsContent/'):
	#f=open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/Buzzfeed/FakeNewsContent/'+str(item))
	print item
	data = json.loads(open('/Users/haritareddy/Desktop/Major_Project/Fake_News_Detection/FakeNewsNet-master/Data/PolitiFact/RealNewsContent/'+str(item)).read())
	#print data
	text=data['text']
	text=re.sub(r'[ \t\n\r]+', ' ', text)
	fake=0
	write_file=open('final_news_content.txt','a+')
	write_file.write(text.encode('utf-8').strip()+'\t'+str(fake)+'\n')