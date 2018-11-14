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


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

#Feature Importance of Top 10 Features

for f in range(0,10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances for the Most Important Features")
plt.bar(range(10), importances[indices[0:10]],
       color="r", align="center")
plt.xticks(range(10), indices[0:10])
plt.xlabel('Feature Number')
plt.ylabel('Importance')
plt.xlim([-1, 10])
plt.show()


#Feature Importance of Bottom 5 Features

for f in range(68,73):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances for the Least Important Features")
plt.bar(range(5), importances[indices[68:73]],
       color="b", align="center")
plt.xticks(range(5), indices[68:73])
plt.xlabel('Feature Number')
plt.ylabel('Importance')

plt.xlim([-1, 5])
plt.show()