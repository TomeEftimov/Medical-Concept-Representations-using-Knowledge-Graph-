from keras.metrics import top_k_categorical_accuracy
from keras import losses
from keras.models import Sequential
from keras.layers import Activation, GRU, LSTM, Dense, Dropout
from keras import regularizers
from keras import backend as K
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import tensorflow as tf 
from numpy import linalg as LA


from proact import proact

def tf_train_health(lookahead, dense, dimension, num_classes, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):

	tf.reset_default_graph()

	timesteps = lookahead-1  

	if dense == True:
		input_dim = dimension
	else:
		input_dim = num_classes
	
	output_dim = num_classes
	LR = 0.01
	beta = 1

	print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + ' hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

	tf_x = tf.placeholder(tf.float32, [None, timesteps, input_dim])                   # (batch, height, width, channel)
	tf_y = tf.placeholder(tf.float32, [None, output_dim])                             # input y

	# RNN
	rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hid_layer)
	outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
	    rnn_cell,                   # cell you have chosen
	    tf_x,                       # input
	    initial_state=None,         # the initial hidden state
	    dtype=tf.float32,           # must given if set initial_state = None
	    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
	)
	output = tf.layers.dense(outputs[:, -1, :], output_dim)              # output based on the last output step
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=output)

	train_op = tf.train.AdamOptimizer(LR).minimize(loss)

	sess = tf.Session()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
	sess.run(init_op)     # initialize var in graph

	for step in range(num_epochs):    # training
		
		for batch in range(batch_size):		
			b_x, b_y = next_batch(x_train, y_train, batch, batch_size)
			_, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})

		preds = sess.run(output, {tf_x: x_test})
		print(accuracy_multilabel(preds, y_test))

	
	preds = sess.run(output, {tf_x: x_test})
	return preds


def next_batch(x,y, batch, batch_size):
	i = batch*batch_size
	j = i + batch_size

	return x[i:j], y[i:j]

def keras_train_health(lookahead, dense, dimension, num_classes, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):

	timesteps = lookahead-1
	input_dim = dimension
	
	print('keras model details : ')
	print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + 'hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()

	if dense == True:
		input_dim = dimension
	else:
		input_dim = num_classes

	output_dim = num_classes

	model.add(GRU(hid_layer,input_shape=(timesteps, input_dim)))  # returns a sequence of vectors of dimension 32
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	model.add(Dense(output_dim, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	            optimizer='adam')

	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=num_epochs)

	preds = model.predict(x_test)
	return preds



def accuracy_multilabel(predict_test, y_test):
	
	acc = 0.0
	total = len(predict_test)

	for i in range(0,total):

		k = max(10,len(set(y_test[i])))
		preds = predict_test[i]
		top_pred = preds.argsort()[-1*k:]
		
		curr_acc = float(len(set(top_pred).intersection(set(y_test[i]) )))/float(len(set(y_test[i])))
		acc = acc + curr_acc
	
	return float(acc)/float(total)



dense_dim = 32
dense = False
num_epochs = 25
batch_size = 64
hid_layer = 16
lookahead = 3
num_classes = 0
dataset = 'mimic'


prep = proact(lookahead=lookahead, dense=dense, dimension=dense_dim)
num_classes = prep.num_classes

x_train, y_train, x_test, y_test = prep.x_train, prep.y_train, prep.x_test, prep.y_test

#tensorflow
preds = tf_train_health(lookahead, dense, dense_dim, num_classes, batch_size, hid_layer, num_epochs, x_train, y_train, x_test, y_test)
print('tf accuracy')
print(accuracy_multilabel(preds, y_test))


#keras
preds = keras_train_health(lookahead, dense, dense_dim, num_classes, batch_size, hid_layer, num_epochs, x_train, y_train, x_test, y_test)
print('keras accuracy')
print(accuracy_multilabel(preds,y_test))










# look_ahead = 5
# data_dim = 26
# timesteps = look_ahead-1


# f_xtr = open("x_train", "r")
# x_train = pkl.load(f_xtr)
# f_xtr.close()


# f_ytr = open("y_train", "r")
# y_train = pkl.load(f_ytr)
# f_ytr.close()


# f_xte = open("x_test", "r")
# x_test = pkl.load(f_xte)
# f_xte.close()


# f_yte = open("y_test", "r")
# y_test = pkl.load(f_yte)
# f_yte.close()


# hid_layer = 512

# # expected input data shape: (batch_size, timesteps, data_dim)
# model = Sequential()
# model.add(LSTM(hid_layer, return_sequences=True,
#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(hid_layer, return_sequences=True))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(hid_layer, return_sequences=True))   # returns a sequence of vectors of dimension 32
# model.add(LSTM(hid_layer))  # return a single vector of dimension 32
# model.add(Dense(data_dim, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# # Generate dummy training data
# #x_train = np.random.random((1000, timesteps, data_dim))
# #y_train = np.random.random((1000, num_classes))

# # Generate dummy validation data
# #x_val = np.random.random((100, timesteps, data_dim))
# #y_val = np.random.random((100, num_classes))

# model.fit(x_train, y_train,
#           batch_size=64, epochs=50,
#           validation_data=(x_test, y_test))