from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()
lem = WordNetLemmatizer()

tsv = 'dataset/finaldataset_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append(ps.stem(lem.lemmatize(ls[0].decode('utf-8'))))
	y_train.append(int(ls[1]))
f.close()

m=len(x_train)

data = x_train

tsv1 = 'dataset/finaldataset_test.txt'
f=open(tsv1,'r')

x_test=[]
y_test=[]

for line in f:
	ls=line.split('\t')
	x_test.append(ps.stem(lem.lemmatize(ls[0].decode('utf-8'))))
	data.append(ps.stem(lem.lemmatize(ls[0].decode('utf-8'))))
	y_test.append(int(ls[1]))
f.close()


print "TfIdf-Vectors"

tvec = TfidfVectorizer(decode_error='ignore')
features2 = tvec.fit_transform(data).toarray()

train_features = []
train_features = features2[:m]
#print(m)
#print(len(y_test))

test_features = []
test_features = features2[m:]
#print(test_features.shape)


ch2 = SelectKBest(chi2, k=25000)
train_chi2 = ch2.fit_transform(train_features, y_train)
test_chi2 = ch2.transform(test_features)

print(test_chi2.shape)



print("################# Naive Bayes Classifier ####################")

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(train_chi2,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_chi2, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_chi2, y_test)

y_pred=clf.predict(test_chi2)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


print ("################### Random Forest Classifier ###############")

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier()
clf.fit(train_chi2,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_chi2, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_chi2, y_test)

y_pred=clf.predict(test_chi2)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)



print ("################### Logistic regression Classifier ###############")

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs')

clf.fit(train_chi2,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_chi2, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_chi2, y_test)

y_pred=clf.predict(test_chi2)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

"""
print ("################### SVM Classifier ###############")

from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf.fit(train_chi2,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_chi2, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_chi2, y_test)

y_pred=clf.predict(test_chi2)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)
"""


