from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

tsv = 'dataset/finaldataset_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append(ls[0])
	y_train.append(int(ls[1]))
f.close()

tsv1 = 'dataset/finaldataset_test.txt'
f=open(tsv1,'r')
x_test=[]
y_test=[]

for line in f:
	ls=line.split('\t')
	x_test.append(ls[0])
	y_test.append(int(ls[1]))
f.close()

print "Number of Training Samples"
print len(x_train)

print "CountVectors"

vectorizer = CountVectorizer()
features1 = vectorizer.fit_transform(x_train).toarray()
#print( vectorizer.vocabulary_ )
print(features1.shape)

"""for f in features1:
	print(f)	
	#print( euclidean_distances(features[0], f) )"""

print "TfIdf-Vectors"

tvec = TfidfVectorizer()
features2 = tvec.fit_transform(x_train).toarray()
print(features2.shape)

"""for f in features2:
	print(f)"""
