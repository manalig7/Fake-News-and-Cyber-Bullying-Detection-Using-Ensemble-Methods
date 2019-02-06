import pandas as pd  
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score, precision_score, f1_score
import pickle
import sys
from bs4 import BeautifulSoup
import re


tsv = '/home/user/Documents/Major Project/Implement-CB/Dataset/finalcb_dataset.txt'
f=open(tsv,'r')


for line in f:
	reslist=[]
	ls=line.split('\t')

	example1 = BeautifulSoup(ls[0], 'lxml')
	string1 = example1.get_text()
	string1 = re.sub(r'@[A-Za-z0-9]+','',string1)
	string1 = re.sub('https?://[A-Za-z0-9./]+','',string1)
	string1 = re.sub("[^a-zA-Z]", " ", string1)
	print (string1+"\t"+ls[1].strip('\n'))
	
f.close()



