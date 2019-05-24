
from keras.models import Sequential
from keras.layers import Activation, GRU, LSTM, Dense
import pandas as pd  
import pickle as pkl
import numpy as np
import random


look_ahead = 5
data_dim = 26
timesteps = look_ahead-1


f_xtr = open("x_train", "rb")
x_train = pkl.load(f_xtr)
f_xtr.close()


f_ytr = open("y_train", "rb")
y_train = pkl.load(f_ytr)
f_ytr.close()


f_xte = open("x_test", "rb")
x_test = pkl.load(f_xte)
f_xte.close()


f_yte = open("y_test", "rb")
y_test = pkl.load(f_yte)
f_yte.close()


hid_layer = 512
print('hidden layer : '+str(hid_layer))

# expected input data shape: (batch_size, timesteps, data_dim)

model = Sequential()
#model.add(GRU(hid_layer,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(Activation('relu'))

model.add(LSTM(hid_layer,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(Activation('relu'))

#model.add(LSTM(hid_layer, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(hid_layer, return_sequences=True))   # returns a sequence of vectors of dimension 32
#model.add(LSTM(hid_layer))  # return a single vector of dimension 32

model.add(Dense(data_dim, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
#x_train = np.random.random((1000, timesteps, data_dim))
#y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
#x_val = np.random.random((100, timesteps, data_dim))
#y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=128, epochs=25,
          validation_data=(x_test, y_test))
