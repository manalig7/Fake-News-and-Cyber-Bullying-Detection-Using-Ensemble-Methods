import pandas as pd  
import numpy as np
np.random.seed(20)
from random import seed
seed(20)
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import warnings 
warnings.filterwarnings(action = 'ignore') 
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import gensim

def fit_transform(d):
	res=[]
	for i in range(0,len(d)):
		temp=[]	
		for j in range(0,len(voc)):
			#print(voc[j])
			if voc[j] in d[i]:
				#print 	(np.mean(model_W2V.wv[voc[j]]))	
				temp.append(np.mean(model_FT.wv[voc[j]]))
			else :
				temp.append(0)		
		res.append(temp)
	return res


x = []
##### training dataset #####

tsv = '/Users/haritareddy/Documents/Method_2_Trial/Major-Project/Fake_News_Detection/Word_Vector_Method_2/finaldataset_train.txt'
f=open(tsv,'r')
y_train=[]
data=[]
lent=[]

tokenizer = RegexpTokenizer(' ', gaps=True)

for line in f :
	ls=line.split('\t')
	x.append(ls[0])
	temp = [] 
	#print(ls[0])
	for j in tokenizer.tokenize(ls[0].decode('utf-8')):
		#print(j) 
	       	temp.append(j) 
	data.append(temp)
	lent.append(len(temp)) 
	y_train.append(int(ls[1]))
f.close()

m=len(x)

#print(max(lent))

##### testing dataset #####
tsv1 = '/Users/haritareddy/Documents/Method_2_Trial/Major-Project/Fake_News_Detection/Word_Vector_Method_2/finaldataset_test.txt'
f=open(tsv1,'r')
y_test=[]

#data1=[]
for line in f:
	ls=line.split('\t')
	x.append(ls[0])
	temp = [] 
	#print(ls[0])
	for j in tokenizer.tokenize(ls[0].decode('utf-8')):
		#print(j) 
	       	temp.append(j) 
	data.append(temp)
	lent.append(len(temp))  
	y_test.append(int(ls[1]))
f.close()

pad_len=max(lent)

model_FT = gensim.models.FastText.load("/Users/haritareddy/Documents/Method_2_Trial/Major-Project/Fake_News_Detection/Word_Vector_Method_2/sg_ft.model")
#model_W2V = Word2Vec(data, size=10, window=5, min_count=1, workers=5, sg=0,max_vocab_size=10000)

#print "SG W2V model_done!"

voc=list(model_FT.wv.vocab)
#	print(len(voc))
XVAL=fit_transform(data)

#print ("Transformed!!")

x_train=[]
x_train=XVAL[:m]
#print(np.array(x_train).shape)
x_test = []
x_test=XVAL[m:]
#print(np.array(x_test).shape)
x_train_sg=x_train
x_test_sg=x_test

##################################################################
##################################################################


#Define the attribute names

#names = ['charcount', 'digratio', 'upratio', 'whiteratio', 'freq_letters', 'freq_sp_char', 'wordcount', 'shortratio', 'word_char_ratio','avg_sentence_length', 'avg_sen_len_words', 'total_diff_words', 'freq_once_ocurring_words', 'freq_twice_ocurring_words', 'word_length_freq_dist', 'num_punc', 'func_count', 'num_lines','num_sentences','num_paras','num_sen_per_para','num_char_per_para','num_word_per_para','greeting','num_quotes','has_url','freq_content_Sp_words','yule_measure','fake']
names=[]
for i in range(0,51):
	names.append(i)
#Create pandas data frame by loading the data from URL

dataframe = pd.read_csv("writeprints_selected_train.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
feature_set = array[:,0:50]
print('Length of feature set')
print(len(feature_set))
print('Length of feature set sg')
print(len(x_train_sg))

dataframe_test = pd.read_csv("writeprints_selected_test.tsv",sep='\t', names=names)
#Create array from data values

array = dataframe_test.values

#Split the data into input and target
#There are 73 features in Writeprints Dataset
feature_set_test = array[:,0:50]


feature_set=feature_set.tolist()
feature_set_test=feature_set_test.tolist()


final_train=[]
final_test=[]

for i in range(len(x_train_sg)):
	print(i)
	final_train.append(x_train_sg[i]+feature_set[i])

for i in range(len(x_test_sg)):
	print(i)
	final_test.append(x_test_sg[i]+feature_set_test[i])

#print(len(final_train[0]))
#print(len(final_test[0]))"""

### Voting method for classification ###

clf1 = GaussianNB()
clf2 = LogisticRegression(random_state=20, solver='lbfgs')
clf3 = RandomForestClassifier(random_state=20)
clf4 = BaggingClassifier(GaussianNB(),max_samples=0.4, max_features=0.8,n_estimators=300,random_state=20)
clf5 = AdaBoostClassifier(n_estimators=300,random_state=20)



print("NB LR RF")
eclf1 = VotingClassifier(estimators=[('nb', clf1), ('lr', clf2), ('rf', clf3)], voting='hard')
# ('mnb', clf1), ('lr', clf2), ('rf', clf3) ,('ada', clf5), ('bag', clf4)
eclf1 = eclf1.fit(final_train, y_train)

print ("\nAccuracy on Training Set :")
print (eclf1.score(final_train, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (eclf1.score(final_test, y_test))

y_pred=eclf1.predict(final_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))

##########################################################
print("NB LR Ada")
eclf1 = VotingClassifier(estimators=[('nb', clf1), ('lr', clf2), ('ada', clf5)], voting='hard')
# ('mnb', clf1), ('lr', clf2), ('rf', clf3) ,('ada', clf5), ('bag', clf4)
eclf1 = eclf1.fit(final_train, y_train)

print ("\nAccuracy on Training Set :")
print (eclf1.score(final_train, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (eclf1.score(final_test, y_test))

y_pred=eclf1.predict(final_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))
###########################################################
print("LR RF Ada")
eclf1 = VotingClassifier(estimators=[('lr', clf2), ('rf', clf3), ('ada', clf5)], voting='hard')
# ('mnb', clf1), ('lr', clf2), ('rf', clf3) ,('ada', clf5), ('bag', clf4)
eclf1 = eclf1.fit(final_train, y_train)

print ("\nAccuracy on Training Set :")
print (eclf1.score(final_train, y_train))

print ("Checking on Test Set")
print ("\nAccuracy on Testing Set :")
print (eclf1.score(final_test, y_test))

y_pred=eclf1.predict(final_test)

print ("\nPrecision Score")
print (precision_score(y_test, y_pred))
print ("\nRecall Score")
print (recall_score(y_test, y_pred))
print ("\nF1 Score")
print (f1_score(y_test, y_pred))