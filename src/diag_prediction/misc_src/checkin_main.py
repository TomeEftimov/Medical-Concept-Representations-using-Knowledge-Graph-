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
import sys
import matplotlib.pyplot as pyplot
from keras.utils import plot_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from checkin_preprocess import Preprocess

class diagnose():

	num_classes = 0 
	lookahead = 0
	x_train, y_train, x_test, y_test = [],[],[],[]

	def tf_train_health_attention(self,lookahead, dense, dimension, num_classes, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):
		
		#only works for lookahead = 2!

		tf.reset_default_graph()

		timesteps = lookahead-1  

		if dense == True:
			input_dim = dimension
		else:
			input_dim = num_classes
		
		output_dim =  num_classes
		LR = 0.01
		beta = 1

		# print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + ' hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

		tf_x = tf.placeholder(tf.float32, [None, timesteps, input_dim])                   # (batch, height, width, channel)
		tf_y = tf.placeholder(tf.float32, [None, output_dim])                             # input y

		# attention mechanism
		att_input = tf_x[:,0,:] #tf.placeholder(tf.float32, [None, timesteps, input_dim])
		att_weights = tf.layers.dense(att_input, input_dim)
		tf_att = tf.multiply(att_weights, att_input)

		# attention mechanism + RNN
		rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hid_layer)
		outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
		    rnn_cell,                   # cell you have chosen
		    tf.expand_dims(tf_att, axis=1),                       # input
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
				b_x, b_y = self.next_batch(x_train, y_train, batch, batch_size)
				_, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})

			preds  = sess.run( output, {tf_x: x_test})
			print(self.accuracy_multilabel(preds, y_test))

		
		preds, att = sess.run([output,att_weights], {tf_x: x_test})

		# print('attention size', att.shape)
		return preds, att

	def tf_train_health(self,lookahead, dense, dimension, num_classes, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):

		tf.reset_default_graph()
		print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + 'hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

		timesteps = lookahead-1  

		if dense == True:
			input_dim = dimension
		else:
			input_dim = num_classes
		
		output_dim =  num_classes
		LR = 0.01
		beta = 1

		# print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + ' hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

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
				b_x, b_y = self.next_batch(x_train, y_train, batch, batch_size)
				_, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})

			preds  = sess.run( output, {tf_x: x_test})
			print(self.accuracy_multilabel(preds, y_test))

		
		preds = sess.run(output, {tf_x: x_test})
		return preds


	def next_batch(self,x,y, batch, batch_size):
		i = batch*batch_size
		j = i + batch_size

		return x[i:j], y[i:j]

	def keras_train_health(self,lookahead, dense, dimension, num_classes, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):

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
		model.add(Activation('relu'))
		model.add(Dropout(0.2))

		model.add(Dense(output_dim, activation='sigmoid'))
		model.compile(loss='binary_crossentropy',
		            optimizer='adam')

		model.fit(x_train, y_train,
		          batch_size=batch_size, epochs=num_epochs)

		preds = model.predict(x_test)
		plot_model(model, to_file='keras_model.png', show_shapes=True)
		SVG(model_to_dot(model).create(prog='dot', format='svg'))

		return preds

	def accuracy_multilabel(self,predict_test, y_test):
		
		acc = 0.0
		total = len(predict_test)

		for i in range(0,total):

			k = max(10,len(set(y_test[i])))
			preds = predict_test[i]
			top_pred = preds.argsort()[-1*k:]
			
			curr_acc = float(len(set(top_pred).intersection(set(y_test[i]) )))/float(len(set(y_test[i])))
			acc = acc + curr_acc
		
		return float(acc)/float(total)

	def accuracy_rare20(self, predict_test, y_test):
		rare20_diag = list([152, 262,  45, 195,  25, 219, 280,  56,  65, 155, 113, 247, 111,
         0, 210, 255, 271, 100,  36, 239,  40,  26, 203,  81, 162,  42,
        24, 141, 249, 120,  39,  74, 182, 201,  35, 234, 125, 276, 275,
        12, 168,  14, 212, 135,  78,  64,  16, 153,  17, 261,  68,  66,
        11, 196,  10,  34, 238])

		k5, total_5, acc_5 = 5, 0.0, 0.0
		k20, total_20, acc_20 = 20, 0.0, 0.0

		for i in range(0, len(predict_test)):
			
			top5 = predict_test[i].argsort()[-1*k5:]
			top20 = predict_test[i].argsort()[-1*k20:]

			check5 = (set(rare20_diag)).intersection(set(y_test[i]))
			check20 = (set(rare20_diag)).intersection(set(y_test[i]))
			
			if len(check5) > 0  :
				total_5 += len(check5)

				if len(check5.intersection(set(top5))) > 0:
					acc_5 += len(check5.intersection(set(top5)))

			if len(check20) > 0:
				total_20 += len(check20)

				if len(check20.intersection(set(top20))) > 0:
					acc_20 += len(check20.intersection(set(top20)))


		if total_5 > 0 :
			acc_5 = acc_5/total_5
		else:
			acc_5 = 0.0
		
		if total_20 > 0 :
			acc_20 = acc_20/total_20
		else:
			acc_20 = 0.0

		return acc_5, acc_20



	def get_names(self,predict_test,y_test):
		firstk = 2
		with open('inv_name','rb') as f:
			inv_name = pkl.load(f)

			sample_truth = y_test[0:firstk]

			k = 10
			sample_preds = [preds.argsort()[-1*k:] for preds in predict_test[0:firstk]]

			name_truth = []
			name_preds = []
			for i in range(0,firstk):
				t = []
				for diag in sample_truth[i]:
					t.append(inv_name[diag])
				name_truth.append(t)

			for i in range(0,firstk):
				t = []
				for diag in sample_preds[i]:
					t.append(inv_name[diag])
				name_preds.append(t)

			print("TRUTH")
			for x in name_truth:
				print(x)

			print("PREDS")
			for x in name_preds:
				print(x)

	def plot_multilabel(self,predict_test, y_test, lookahead):
		
		acc = 0.0
		k = 10
		total = len(predict_test)

		x_axis = []
		y_axis = []

		for i in range(0,total):

	#		predict_test[i] vs y_test[i]
			k = len(set(y_test[i]))

			preds = predict_test[i]
			top_pred = preds.argsort()[-1*k:]
			
			# trueL = []
			
			# for k in top_pred:
			# 	if trueL[k] == 1:	
			# 		score = score + 1
			# 		break

			curr_acc = (float(len(set(top_pred).intersection(set(y_test[i]) )))/float(len(set(y_test[i]))))
			acc = acc + curr_acc

			x_axis.append(lookahead)
			y_axis.append(curr_acc)

			# print((float(len(set(top_pred).intersection(y_test[i])))/float(len(y_test[i]))))
			# break

		# print('SCORE '+str(score))
		

		# pyplot.plot(x_axis,y_axis,'bo')
		# pyplot.show()

		return float(acc)/float(total)				

	def compile(self,dataset,lookahead, dense, dimension):

		prep = Preprocess(dataset=dataset,lookahead=lookahead, dense=dense, dimension=dimension)
		num_classes = prep.num_classes

		x_train, y_train, x_test, y_test = prep.x_train, prep.y_train, prep.x_test, prep.y_test

		return x_train, y_train, x_test, y_test, num_classes

	def __init__(self,dataset='mimic',lookahead=3, dense=False, dimension=32 ):

		num_epochs = 20
		batch_size = 64
		hid_layer = 64
		self.lookahead = lookahead
		# dataset = 'proact'
		dataset = 'mimic'

		self.x_train, self.y_train, self.x_test, self.y_test, self.num_classes  = self.compile(dataset=dataset, lookahead=lookahead, dense=dense, dimension= dimension)


	def run_model(self, lookahead=3, dense=False, dimension=32,  num_classes=None, batch_size=64, hid_layer=64, num_epochs=20, x_train = None, y_train = None, x_test = None, y_test = None ):

		if num_classes == None:
			num_classes = self.num_classes
		if x_train == None:
			x_train = self.x_train
		if y_train == None:
			y_train = self.y_train
		if x_test == None:
			x_test = self.x_test
		if y_test == None:
			y_test = self.y_test

		lookahead = self.lookahead


		#tensorflow
		print('tf accuracy without attention : ')
		preds = self.tf_train_health(lookahead, dense, dimension,  num_classes, batch_size, hid_layer, num_epochs, x_train, y_train, x_test, y_test)
		att = None
		accuracy = self.accuracy_multilabel(preds, y_test)
		print(accuracy)


		#tensorflow + attention on 284 diagnosis codes ( valid only when lookahead = 2 )
		print('tf accuracy with attention')
		preds, att = self.tf_train_health_attention(lookahead, dense, dimension,  num_classes, batch_size, hid_layer, num_epochs, x_train, y_train, x_test, y_test)
		accuracy = self.accuracy_multilabel(preds, y_test)
		print(accuracy)

		# self.plot_multilabel(preds, y_test,lookahead)
		# print(self.accuracy_rare20(preds, y_test))
		# get_names(preds,y_test)

		return accuracy, att, preds

		#keras
		# preds = self.keras_train_health(lookahead, dense, dimension, num_classes, batch_size, hid_layer, num_epochs, x_train, y_train, x_test, y_test)
		# print('keras accuracy')
		# keras_accuracy = self.accuracy_multilabel(preds,y_test)
		# print(keras_accuracy)

		# accuracy = keras_accuracy

		# return accuracy


lookahead_array = [2]
dimension_array = [32]
dense = False  #False indicates multihot, True indicates dense
dimension = 64 # indicates dense dimension (relevant when dense is True)

for lookahead in lookahead_array:
	for dim in dimension_array:
		print('lookahead ',lookahead)
		d = diagnose(lookahead=lookahead, dimension=dimension, dense=dense)
		accuracy, att, preds = d.run_model(lookahead=lookahead, dimension = dimension, dense = dense, hid_layer=32, batch_size=128, num_epochs=20)

# pyplot.savefig('sliding.png')

