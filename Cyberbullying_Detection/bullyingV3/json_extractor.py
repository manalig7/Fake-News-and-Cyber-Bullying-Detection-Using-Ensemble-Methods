import json

f=open('tweet_text','w')

with open('tweet.json') as json_file:  
    data = json.load(json_file)
    print "Recahed Here"
    print data
    for p in data['text']:
    	print p
    	f.write(p+'\n')
        