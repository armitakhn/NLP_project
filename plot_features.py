# import time to find consuming steps
import time
start = time.time()

# utility libraries
import numpy as np
import pandas as pd
import csv

# plotting stuffs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb

end = time.time()
print('Loading libraries takes %.4f s' % (end-start))


##################################
# ====== reading features ====== #
##################################
start = time.time()

path_data = 'data/'

# get the features
orig_training_features = np.genfromtxt(path_data + 'training_features.csv', delimiter=',',skip_header=1)
training_features = np.nan_to_num(orig_training_features)

# get the labels
training = np.genfromtxt(path_data + 'training_set.txt', dtype=str) # to get the label only
labels = training[:, 2]

end = time.time()
print('Loading features takes %.4f s' % (end-start))


###############################
# ====== PCA reduction ====== #
###############################
start = time.time()

from sklearn.decomposition import PCA

my_pca = PCA(n_components=2) # dimensionality reduction
ft_pca = my_pca.fit_transform(training_features)

end = time.time()
print('PCA reduction takes %.4fs' % (end-start))

idx_neg = (labels == '0')
idx_pos = (labels == '1')
neg_class = ft_pca[idx_neg]
pos_class = ft_pca[idx_pos]

x1_neg = neg_class[:, 0]
x2_neg = neg_class[:, 1]
x1_pos = pos_class[:, 0]
x2_pos = pos_class[:, 1]

plt.scatter(x1_neg, x2_neg, color='b')
plt.scatter(x1_pos, x2_pos, color='r')
plt.show()