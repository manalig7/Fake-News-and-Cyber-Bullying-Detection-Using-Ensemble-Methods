import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features

names=[]

for i in range(0,75):
  names.append(i)
print("names done")

#names = ['charcount', 'digratio', 'upratio', 'whiteratio', 'freq_letter_a', 'freq_letter_b','freq_letter_c','freq_letter_d','freq_letter_e','freq_letter_f','freq_letter_g','freq_letter_h','freq_letter_i','freq_letter_j','freq_letter_k','freq_letter_l','freq_letter_m','freq_letter_n','freq_letter_o','freq_letter_p','freq_letter_q','freq_letter_r','freq_letter_s','freq_letter_t','freq_letter_u','freq_letter_v','freq_letter_w','freq_letter_x','freq_letter_y','freq_letter_z','freq_sp_char', 'word_count', 'short_ratio', 'char_per_word','avg_word_length','avg_sentence_length_char', 'avg_sen_len_words', 'total_diff_words', 'freq_once_ocurring_words', 'freq_twice_ocurring_words', 'word_length_freq_dist', 'num_punc', 'func_count', 'num_lines','num_sentences','num_paras','num_sen_per_para','num_char_per_para','num_word_per_para','greeting','num_quotes','has_url','freq_content_Sp_words','yule_measure']
dataframe = pd.read_csv("writeprints_features_final.tsv",sep='\t', names=names)
print("file read")
#dataframe.dropna(axis=0,inplace=True)


array = dataframe.values

X = array[:,0:73]

y = array[:,73]

num_quoted_real=0
num_quoted_fake=0
num_fake=0
num_real=0

for i in range(0,len(X)):
	if int(y[i])==1:
		num_fake=num_fake+1
		num_quoted_fake=num_quoted_fake+float(X[i][69])
	elif int(y[i])==0:
		num_real=num_real+1
		num_quoted_real=num_quoted_real+float(X[i][69])

fake_ratio=float(num_quoted_fake)/num_fake
real_ratio=float(num_quoted_real)/num_real

x=["Fake","Real"]
y=[fake_ratio,real_ratio]

plt.figure()
plt.title("Quoted Content for Real and Fake News")
plt.bar(range(2), y,
       color="g", align="center")
plt.xticks(range(2), x)
plt.xlabel('News Type')
plt.ylabel('Average Quoted Content')

plt.show()



