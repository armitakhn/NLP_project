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
training_features = np.genfromtxt(path_data + 'training_features.csv', delimiter=',',skip_header=1)

end = time.time()
print('Loading features takes %.4f s' % (end-start))


###############################
# ====== PCA reduction ====== #
###############################
start = time.time()

from sklearn.decomposition import PCA

my_pca = PCA(n_components=2) # dimensionality reduction
ft_pca = my_pca.fit_transform(training_features[0:1000])

end = time.time()
print('PCA reduction takes %.4fs' % (end-start))


####################################
# ====== Plotting with TSNE ====== #
####################################
start = time.time()

from sklearn.manifold import TSNE

my_tsne = TSNE(n_components=2)
ft_tsne = my_tsne.fit_transform(ft_pca)

# plotting real pretty stuffs
fig, ax = plt.subplots()
ax.scatter(ft_tsne[:,0], ft_tsne[:,1],s=3)
fig.suptitle('t-SNE visualization of word embeddings',fontsize=20)
fig.set_size_inches(11,7)
fig.show()
plt.show()

end = time.time()
print('TSNE processing takes %.4fs' % (end-start))