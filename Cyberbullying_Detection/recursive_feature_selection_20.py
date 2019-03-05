"""
This file contains the code for feature selecion for writeprints dataset as well as the application of the random forest classifier

"""


#Import the required packages

#Import pandas to read csv import pandas

#Import numpy for array related operations import numpy

#Import sklearn's feature selection algorithm

from sklearn.feature_selection import SelectKBest
import pandas as pd  
import numpy as np
import sys
#Import chi2 for performing chi square test 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#URL for loading the dataset

np.random.seed(20)
#Define the attribute names

#names = ['charcount', 'digratio', 'upratio', 'whiteratio', 'freq_letters', 'freq_sp_char', 'wordcount', 'shortratio', 'word_char_ratio','avg_sentence_length', 'avg_sen_len_words', 'total_diff_words', 'freq_once_ocurring_words', 'freq_twice_ocurring_words', 'word_length_freq_dist', 'num_punc', 'func_count', 'num_lines','num_sentences','num_paras','num_sen_per_para','num_char_per_para','num_word_per_para','greeting','num_quotes','has_url','freq_content_Sp_words','yule_measure','fake']
names=[]
for i in range(0,98):
	names.append(i)
print("names done")
#Create pandas data frame by loading the data from URL

dataframe = pd.read_csv("Writeprints_Lower_Sentiment/finalcb_spell_check_with_sentiment_features_train.tsv",sep='\t', names=names)
print("file read")
#Create array from data values

array = dataframe.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
X = array[:,0:97]

Y = array[:,97]
print("attributes Split")

model = LogisticRegression(random_state=np.random.seed(20))
#print X[0], Y[0]
# create the RFE model and select 3 attributes
rfe = RFE(model, 80)
rfe = rfe.fit(X, Y)
# summarize the selection of the attributes


#Summarize scores 

np.set_printoptions(precision=3)
print("-------------")

print("summarized scores")

print(rfe.support_)

selected_feature_names=[]

for i in range(0,len(rfe.support_)):
	if rfe.support_[i]==True:
		selected_feature_names.append(i)

"""
print "Selected Features"
for i in range(0,len(selected_feature_names)):
	print selected_feature_names[i], selected_feature_scores[i]
"""
#We have the selected feature names now
feature_set_temp=X

feature_set_temp=np.ndarray.tolist(feature_set_temp)

y_train=Y
feature_set=[]
for i in range(0,len(X)):
	feature_set.append([])

#We need to remove the undesired features from the feature set
for item in selected_feature_names:
	for j in range(0,len(feature_set)):
		feature_set[j].append(feature_set_temp[j][item])

names=[]
for i in range(0,98):
	names.append(i)

dataframe1 = pd.read_csv("Writeprints_Lower_Sentiment/finalcb_spell_check_segment_with_sentiment_features_test.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe1.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset

X_test = array[:,0:97]

Y_test = array[:,97]

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


#Implementation of Random Forest Classifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

clf_rand = RandomForestClassifier(random_state=np.random.seed(20))
clf_multi_nb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
clf_gaussian_nb = GaussianNB()
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_svm=svm.SVC(gamma='auto',C=0.80,random_state=np.random.seed(20))
clf_log= LogisticRegression(random_state=np.random.seed(20), solver='lbfgs',multi_class='multinomial')




clf_gbc_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5),random_state=np.random.seed(20),
                            max_samples=0.4, max_features=0.8,n_estimators=300)
clf_gbc_rf = BaggingClassifier(RandomForestClassifier(random_state=np.random.seed(20)),random_state=np.random.seed(20),
                            max_samples=0.4, max_features=0.8,n_estimators=300)
clf_extra = ExtraTreesClassifier(n_estimators=200, max_depth=None,min_samples_split=2, random_state=np.random.seed(20))
clf_ada= AdaBoostClassifier(n_estimators=300,random_state=np.random.seed(20))
clf_grad_boost = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000,max_depth=3, min_samples_split=5, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=np.random.seed(20))
"""

print "BASIC CLASSIFIER RESULTS"

############## RANDOM FOREST #####################
print "################## RANDOM FOREST #####################"
clf_rand.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_rand.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_rand.score(feature_set_test, y_test))

y_pred=clf_rand.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

############## MULTI NB #####################
print "################## MULTI NB #####################"
clf_multi_nb.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_multi_nb.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_multi_nb.score(feature_set_test, y_test))

y_pred=clf_multi_nb.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

############## GAUSSIAN NB #####################
print "################## GAUSSIAN NB #####################"
clf_gaussian_nb.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_gaussian_nb.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_gaussian_nb.score(feature_set_test, y_test))

y_pred=clf_gaussian_nb.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

############## KNN #####################
print "################## KNN #####################"
clf_knn.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_knn.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_knn.score(feature_set_test, y_test))

y_pred=clf_knn.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))
############## SVM #####################
print "################## SVM #####################"
clf_svm.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_svm.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_svm.score(feature_set_test, y_test))

y_pred=clf_svm.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))
"""
############## LOGISTIC REGRESSION #####################
print "################## LOGISTIC REGRESSION #####################"
clf_log.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_log.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_log.score(feature_set_test, y_test))

y_pred=clf_log.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))
"""
print "***********************************************************"
print "******************ENSEMBLE CLASSIFIERS********************"
print "***********************************************************"

print "################## GBC WITH KNN #####################"
clf_gbc_knn.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_gbc_knn.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_gbc_knn.score(feature_set_test, y_test))

y_pred=clf_gbc_knn.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

print "################## GBC WITH RF #####################"
clf_gbc_rf.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_gbc_rf.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_gbc_rf.score(feature_set_test, y_test))

y_pred=clf_gbc_rf.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

print "################## EXTRA TREES #####################"
clf_extra.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_extra.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_extra.score(feature_set_test, y_test))

y_pred=clf_extra.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

print "################## ADABOOST #####################"
clf_ada.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_ada.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_ada.score(feature_set_test, y_test))

y_pred=clf_ada.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

print "################## GRADIENT BOOSTING #####################"
clf_grad_boost.fit(feature_set,y_train)
print ("\nAccuracy on Training Set :")
print (clf_grad_boost.score(feature_set, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (clf_grad_boost.score(feature_set_test, y_test))

y_pred=clf_grad_boost.predict(feature_set_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))
"""
