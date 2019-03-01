from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score

tsv = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_clean_train.txt'
f=open(tsv,'r')
x_train=[]
y_train=[]

for line in f:
	ls=line.split('\t')
	x_train.append(ls[0])
	y_train.append(int(ls[1]))
f.close()

m=len(x_train)

data = x_train

tsv1 = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_clean_test.txt'
f=open(tsv1,'r')

x_test=[]
y_test=[]

for line in f:
	ls=line.split('\t')
	x_test.append(ls[0])
	data.append(ls[0])
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
print(test_features.shape)



print("################# Naive Bayes Classifier ####################")

from sklearn.naive_bayes import MultinomialNB


clf = MultinomialNB()
clf.fit(train_features,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_features, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_features, y_test)

y_pred=clf.predict(test_features)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)


print ("################### Random Forest Classifier ###############")

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(random_state=20)
clf.fit(train_features,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_features, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_features, y_test)

y_pred=clf.predict(test_features)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

print ("################### Logistic regression Classifier ###############")

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=20, solver='lbfgs')

clf.fit(train_features,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_features, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_features, y_test)

y_pred=clf.predict(test_features)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

"""
print ("################### SVM Classifier ###############")

from sklearn.svm import SVC

clf = SVC(gamma='auto',random_state=20)
clf.fit(train_features,y_train)

print "\nAccuracy on Training Set :"
print clf.score(train_features, y_train)

print "Checking on Test Set"
print "\nAccuracy on Testing Set :"
print clf.score(test_features, y_test)

y_pred=clf.predict(test_features)

print "\nPrecision Score"
print precision_score(y_test, y_pred)
print "\nRecall Score"
print recall_score(y_test, y_pred)
print "\nF1 Score"
print f1_score(y_test, y_pred)

"""




