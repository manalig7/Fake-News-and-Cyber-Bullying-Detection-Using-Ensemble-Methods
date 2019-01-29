from textblob import TextBlob
import pandas as pd  
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import pickle
import sys


tsv = 'wikidata.txt'
f=open(tsv,'r')
#x_train=[]
#y_train=[]

for line in f:
	ls=line.split('\t')
	#x_train.append(ls[0])
	#y_train.append(int(ls[1]))

	testimonial = TextBlob(ls[0].decode('utf-8'))
	print testimonial.sentiment
	print ls[1]

f.close()
