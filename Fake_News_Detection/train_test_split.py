
import pandas as pd  
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import pickle
import sys

tsv = 'FakeNewsNet_Dataset/fakenewsnet.txt'
f=open(tsv,'r')
x=[]
y=[]

for line in f:
	ls=line.split('\t')
	x.append(ls[0])
	y.append(int(ls[1]))


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

p1=open('FakeNewsNet_Dataset/fakenewsnet_train.txt','w')

for i in range(0,len(x_train)):
	p1.write(x_train[i]+'\t'+str(y_train[i])+'\n')

p2=open('FakeNewsNet_Dataset/fakenewsnet_test.txt','w')

for i in range(0,len(x_test)):
	p2.write(x_test[i]+'\t'+str(y_test[i])+'\n')