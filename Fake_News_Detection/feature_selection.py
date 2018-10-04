# Feature Extraction Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
f=open('writeprints_features.tsv','r')

X=[]
Y=[]
for line in f:
	ls=line.split('\t')
	n=len(ls)
	Y.append(int(ls[n-1]))
	X.append([])
	length=len(X)
	for i in range(0,n-1):
		X[length-1].append(float(ls[i]))


test = SelectKBest(score_func=chi2, k=30)
fit = test.fit(X, Y)

numpy.set_printoptions(precision=3)

features = fit.transform(X)
# summarize selected features

print "Selected Features"
print len(features)
print len(features[0])
#print features[0][0]