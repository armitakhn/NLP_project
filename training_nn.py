import csv
import json
import numpy as np

import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

from gensim.models.word2vec import Word2Vec

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


####################
# parameter values #
####################

path_data = 'data/'
path_submission = 'submission/'

#nb_features = len(features) # number of features
stpwd_thd = 10 # number of most frequent words to consider as stopwords
do_static = False # non static CNN
nb_filters = 100 # number of filters
filter_size = 3 # size of each filter --> 3x3 matrix
drop_rate = 0.3 # dropout rate
batch_size = 64
nb_epoch =  3 # number of episodes
my_optimizer = 'adam' # use 'adam' of sklearn as optimizer
my_patience = 1 # for early stopping strategy


print('\n=== parameter values: ===')
print('non-static:',do_static)
print('number of filters applied to each region:',nb_filters)
print('filter size:',filter_size)
print('dropout rate:',drop_rate)
print('batch size:',batch_size)
print('number of epochs:',nb_epoch)
print('optimizer:',my_optimizer)
print('patience:',my_patience)
print('=== end parameter values ===')


######################
# load training data #
######################

training = np.genfromtxt(path_data + 'training_set.txt', dtype=str)

print('\nTraining data loaded:', training.shape)


###############
# load labels #
###############

labels = np.array(training[:, 2]) # get the labels

print('\nLabels loaded:', labels.shape)


##########################
# load training features #
##########################

training_features = np.array(np.genfromtxt(path_data + 'training_features.csv', delimiter=',',skip_header=1))

print('\nTraining features loaded:', training_features.shape)


#########################
# load testing features #
#########################

testing_features = np.array(np.genfromtxt(path_data + 'testing_features.csv', delimiter=',',skip_header=1))

print('\nTesting features loaded:', testing_features.shape)



################
# defining CNN #
################

nb_features = training_features.shape[1]

print('\nDefining CNN...')

my_input = Input(shape=(nb_features,))


embedding = Embedding(
  input_dim = nb_features+1,
  output_dim = 20,
  trainable = do_static,
  ) (my_input)

embedding_dropped = Dropout(drop_rate)(embedding) # apply dropout

conv = Conv1D(
  filters = nb_filters,
  kernel_size = filter_size,
  activation = 'relu',
  )(embedding_dropped)

pooled_conv = GlobalMaxPooling1D()(conv)

pooled_conv_dropped = Dropout(drop_rate)(pooled_conv)

# we finally project onto a single unit output layer with relu activation
prob = Dense(1, activation='relu')(pooled_conv_dropped)

model = Model(my_input, prob)

# We add a vanilla hidden layer:
# model.add(Dense(hidden_dims))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))

model.compile(optimizer = my_optimizer, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

print('\nModel compiled')

# # you can access the layers of the model with model.layers
# # then the input/output shape of each layer with, e.g., model.layers[0].input_shape or model.layers[0].output_shape

# # nice as well:
model.summary()

print('\nTotal number of model parameters:',model.count_params())


################
# training CNN #
################

# ~160s/epoch on 4-core i7 CPU (on GPU much faster). Gets to 0.87 after 2 epochs
# warning: on CPU, by default, will use all the cores of the machine

early_stopping = EarlyStopping(monitor = 'val_acc', patience = my_patience, mode = 'max')

# use the .fit() method with appropriate values for the arguments 'x', 'y', 'batch_size', 'epochs', 'validation_data', and 'callbacks'. 'validation_data' needs to be passed as a tuple, 'callbacks' as a list
model.fit(
  x = training_features, 
  y = labels, 
  batch_size = batch_size,
  epochs = nb_epoch, 
  shuffle = True,
  callbacks = [early_stopping]
)

# # to reload model:
# # model = load_model(path_to_save + name_save) # I do not have hdf5 module


#######################################
# identifying predictive text regions #
#######################################

# the feature maps that we find at the output of the convolutional layer, before pooling is applied
# provide region embeddings (in an nb_filters-dimensional space)
# there are (max_size-filter_size+1) regions of size filter_size in an input of size max_size

predictions = model.predict(testing_features).tolist()

filename = 'submission_cnn_01.csv'

# write submission to file
with open(path_submission + filename, 'wb') as f:
  csv_out = csv.writer(f)
  csv_out.writerow(['id','category'])
  for row in predictions:
    csv_out.writerow(row)


# predictions are probabilities of belonging to class 1

# get_softmax = K.function([model.layers[0].input,K.learning_phase()], [model.layers[6].output])
# predictions = get_softmax([np.array(x_test[:100]),0])[0]

# # note: you can also use directly: predictions = model.predict(x_test[:100]).tolist()