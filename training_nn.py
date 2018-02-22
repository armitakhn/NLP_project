##################################
# ====== importing stuffs ====== #
##################################
import time

start = time.time()

import numpy as np

from sklearn import model_selection as ms

import os
import csv

import tensorflow
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Conv1D, MaxPooling1D
from keras import activations

end = time.time()

print("tensorflow version %s (should be at least 0.12.1)" % tensorflow.__version__)
print("keras version %s (should be at least 2.0.7)" % keras.__version__)
print('loading libraries takes %.4fs' % (end-start))


def write_submission(filename, pred):
  '''
  Write prediction result in a submission file

  Parameters
  ----------
  filename: name of submission file
  pred: prediction array
    
  '''
  prediction = []

  idx = 0
  for p in pred:
    choice = np.argmax(p) # should be either 0 or 1
    prediction.append((idx, choice))
    idx += 1

  with open(path_submission + filename, 'w', newline='') as f:
    csv_out = csv.writer(f)
    csv_out.writerow(['id','category'])
    for row in prediction:
      csv_out.writerow(row)

##############################
# ====== loading data ====== #
##############################
start = time.time()

path_data = 'data/'
path_submission = 'submission/'

# Number of classes: binary --> YES/NO if there is an edge
num_classes = 2

# load training features (training data)
features = np.genfromtxt(path_data + 'training_features.csv', delimiter=',',skip_header=1)
training = np.genfromtxt(path_data + 'training_set.txt', dtype=str) # to get the label only
labels = training[:, 2]

x_train, x_test, y_train, y_test = ms.train_test_split(features, labels, test_size=0.35)

input_shape = (x_train.shape[1], 1) # nb of features

# load training & testing labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# ====== loading features for prediction ====== #
x_pred = np.genfromtxt(path_data + 'testing_features.csv', delimiter=',',skip_header=1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
x_pred = x_pred.astype('float32')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
print('x_pred shape:', x_pred.shape)

end = time.time()
print('loading data takes %.4fs' % (end-start))


######################################
# ====== building the network ====== #
######################################
start = time.time()

 
# ====== FF-NN ====== #
model_nn = Sequential()

model_nn.add(Flatten(input_shape=input_shape))

# use with Leaky Relu
model_nn.add(Dense(30, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))

model_nn.add(Dense(40, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))

model_nn.add(Dense(50, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))
model_nn.add(Dropout(rate=0.25))

model_nn.add(Dense(60, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))

model_nn.add(Dense(50, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))

model_nn.add(Dense(40, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))

model_nn.add(Dense(30, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))
model_nn.add(Dropout(rate=0.25))

model_nn.add(Dense(20, kernel_initializer='lecun_uniform'))
model_nn.add(LeakyReLU(0.25))


## used with ReLU
# model_nn.add(Dense(30, activation='relu'))
# model_nn.add(Dense(40, activation='relu'))
# model_nn.add(Dense(50, activation='relu'))
# model_nn.add(Dense(60, activation='relu'))  
# model_nn.add(Dense(50, activation='relu'))
# model_nn.add(Dense(40, activation='relu'))
# model_nn.add(Dense(30, activation='relu'))
# model_nn.add(Dense(20, activation='relu'))

model_nn.add(Dense(num_classes, activation='softmax'))
# ====== FF-NN ====== #


# ====== CNN ====== #
# model_cnn = Sequential()

# model_cnn.add(Conv1D(64, (3), activation='relu', input_shape=input_shape))
# model_cnn.add(MaxPooling1D(pool_size=2))

# model_cnn.add(Conv1D(32, (3), activation='relu'))
# model_cnn.add(MaxPooling1D(pool_size=2))

# model_cnn.add(Dropout(rate=0.25))
# model_cnn.add(Dense(16, activation='relu'))

# model_cnn.add(Dropout(rate=0.35))
# model_cnn.add(Flatten())
# model_cnn.add(Dense(num_classes, activation='softmax'))
# ====== CNN ====== #

model_nn.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

model_nn.summary()


batch_size = 32
epochs = 40

early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=2)

# Run the train
history = model_nn.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test,y_test),
                        callbacks=[early_stopping]
                        )

# Predict
pred = model_nn.predict(x_pred, verbose=1)

write_submission('submission_ffnn_02.csv', pred)