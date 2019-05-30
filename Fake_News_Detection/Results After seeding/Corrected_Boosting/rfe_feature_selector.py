from random import seed
import numpy as np
seed(20)
np.random.seed(20)
from sklearn.feature_selection import SelectKBest
import pandas as pd  
import gensim

import sys
#Import chi2 for performing chi square test 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
##########################FOR SQ W2V###########################
#############################################################
import numpy as np
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score


names=[]
for i in range(0,74):
	names.append(i)
#Create pandas data frame by loading the data from URL

dataframe = pd.read_csv("writeprints_features_final.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
X = array[:,0:73]

Y = array[:,73]

#np.random.seed(20)
#model = LogisticRegression(np.random.seed(20))
#model=pickle.load(open('model_feature_selection', 'rb'))
#pickle.dump(model, open('model_feature_selection', 'wb'))
model = LogisticRegression(random_state=20)
# create the RFE model and select 3 attributes
rfe = RFE(model, 50)
rfe = rfe.fit(X, Y)

rankings= list(rfe.ranking_)
#print rankings.count(1)
#print rfe.support_
# summarize the selection of the attributes

np.set_printoptions(precision=3)


selected_feature_names=[]



for i in range(0,len(rfe.support_)):
	if rfe.support_[i]==True:
		selected_feature_names.append(i)

print selected_feature_names

"""
p=open('selected_features','w')

for item in selected_feature_names:
	p.write(str(item)+'\n')

print selected_feature_names
"""
"""
p=open('selected_features','r')

for line in p:
	ls=line.split()
	selected_feature_names.append(int(ls[0]))
"""

feature_set_temp=X

feature_set_temp=np.ndarray.tolist(feature_set_temp)


feature_set=[]
for i in range(0,len(X)):
	feature_set.append([])

#We need to remove the undesired features from the feature set
for item in selected_feature_names:
	for j in range(0,len(feature_set)):
		feature_set[j].append(feature_set_temp[j][item])

names=[]
for i in range(0,74):
	names.append(i)

dataframe1 = pd.read_csv("writeprints_features_final_test.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe1.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset

X_test = array[:,0:73]

Y_test = array[:,73]

feature_set_test_temp=X_test

feature_set_test_temp=np.ndarray.tolist(feature_set_test_temp)

y_test=Y_test
feature_set_test=[]
for i in range(0,len(X_test)):
	feature_set_test.append([])

#We need to remove the undesired features from the feature set
for item in selected_feature_names:
	for j in range(0,len(feature_set_test)):
		feature_set_test[j].append(feature_set_test_temp[j][item])


f1=open('writeprints_features_selected_train.tsv','w')

for i in range(0,len(feature_set)):
	for j in range(0,len(feature_set[i])):
		f1.write(str(feature_set[i][j])+'\t')
	f1.write(str(Y[i])+'\n')

f1.close()


f2=open('writeprints_features_selected_test.tsv','w')

for i in range(0,len(feature_set_test)):
	for j in range(0,len(feature_set_test[i])):
		f2.write(str(feature_set_test[i][j])+'\t')
	f2.write(str(Y_test[i])+'\n')

f2.close()














