from sklearn.feature_selection import SelectKBest
import pandas as pd  
import numpy as np
import sys
from sklearn.feature_selection import chi2

names=[]
for i in range(0,74):
	names.append(i)
print("names done")

dataframe = pd.read_csv("writeprints_features_final.tsv",sep='\t', names=names)
print("file read")

selected_feature_names=[]
selected_feature_scores=[]

array = dataframe.values

X = array[:,0:73]

Y = array[:,73]
print("Attributes Split")

test = SelectKBest(score_func=chi2, k=5)
print("Selected K Best")

fit = test.fit(X, Y)

np.set_printoptions(precision=3)
print("-------------")

print("Summarized Scores")

print(fit.scores_)

fit.scores_=np.ndarray.tolist(fit.scores_)

while len(selected_feature_names)<10: #Selecting 35 features
	maxi=-1
	index=-1
	for i in range(0,len(fit.scores_)):
		if fit.scores_[i]>maxi:
			maxi=fit.scores_[i]
			index=i
	selected_feature_names.append(names[index])
	selected_feature_scores.append(fit.scores_[index])
	names.pop(index)
	fit.scores_.pop(index)

print "Selected Features"
for i in range(0,len(selected_feature_names)):
	print selected_feature_names[i], selected_feature_scores[i]