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
import matplotlib.pyplot as pyplot


#from mimic3_preprocess import unique_icd9_codes, count_icd9_codes, inv_idx, x_train, y_train, x_test, y_test
#from m3_preprocess_coG import unique_icd9_codes, count_icd9_codes, inv_idx, x_train, y_train, x_test, y_test
# from hadm_preprocess import Preprocess
from var_preprocess import Preprocess
#from hadm_preprocess import PreprocessMulti

def codes(b1, b2):
	return [code for code, count in count_icd9_codes.items() if (count >= b1 and count < b2)  ]

def get_rare():
	
	#obtain a list of diagnosis which are rare as a list	
	code_cts = list(count_icd9_codes.values())
	code_cts.sort()

	total_codes = len(code_cts)

	boundary = []
	for i in range(0,5):
		boundary.append(code_cts[int(i*total_codes/5)])

	boundary.append(code_cts[total_codes-1]+1)
	
	top = []
	for i in range(0,5):
		top.append(codes(boundary[i], boundary[i+1]))

	return top

def accuracy_5(labels,results):
	num_test = len(labels)
	num_classes = len(labels[0])

	classes = []

	k = 5
	acc_count = [0]*5
	t_count = [0]*5

	for i in range(0,num_test):
		
		r = results[i]
		l = labels[i]
		top_k_r = r.argsort()[-1*k:][::-1]
		top_k_r = [code_dict[t] for t in top_k_r]

		idx = [idx for idx in range(0,len(l)) if l[idx] == 1 ][0]
		code = code_dict[idx]
		
		for it in range(0,5):
			if code in top[it]:
				t_count[it] = t_count[it]+1
				if code in top_k_r:
					acc_count[it] = acc_count[it]+1

	acc = {}
	for i in range(0,5):
		acc[str(20*i)+'-'+str(20*(i+1))] = float(acc_count[i])/float(t_count[i]+1)

	return acc

def h_reg(weight_matrix):

	sum = 0.0

	sum = sum + K.sum(0.01*K.abs(weight_matrix))
	# for i in range(0,282):
	# 	for j in range(i+1, 282):
	# 		sum = sum + 0.01 * tf.norm(tf.subtract(tf.gather(weight_matrix, i), tf.gather(weight_matrix, j)))
	
	return sum

	# tf.subtract()
	# print(tf.gather(weight_matrix,i))
	# l1_sum = 0.0
	# for x in weight_matrix:
	# 	for y in x:
	# 		l1_sum = l1_sum+abs(x)

	# return 0.01*K.sum(K.abs(l1_sum))
# from keras.regularizers import l2
# model.add(Dense(64, 64, W_regularizer = l2(.01)))

def regularizer(weight, num_hid):

	l2 = tf.constant(0.0, tf.float32)
	rows = num_hid

	# l2 = tf.linalg.norm(tf.subtract(weight[1],weight[7]))
	for i in range(0, 284):
		for j in range(i+1,284):
			l2 = tf.add(l2, tf.linalg.norm(tf.subtract(weight[:,i],weight[:,j])))

	return l2


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 5) 

def tf_train_health(dense, dimension, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test, train_seq, test_seq):

	tf.reset_default_graph()
	# look_ahead = 3
	# timesteps = look_ahead-1
	input_dim = 284
	output_dim = 284
	LR = 0.01
	beta = 1

	print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + 'hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

	tf_x = tf.placeholder(tf.float32, [None, 11, input_dim])                   # (batch, height, width, channel)
	tf_y = tf.placeholder(tf.float32, [None, output_dim])                             # input y
	seqlen = tf.placeholder(tf.int32,[None])
	# RNN


	rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hid_layer)
	outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
	    rnn_cell,                   # cell you have chosen
	    tf_x,        
	    sequence_length = seqlen,              # input
	    initial_state=None,         # the initial hidden state
	    dtype=tf.float32,           # must given if set initial_state = None
	    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
	)
	output = tf.layers.dense(outputs[:, -1, :], output_dim)              # output based on the last output step
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=output)

	# loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
	# reg =  beta * regularizer(tf.trainable_variables(scope='dense/kernel:0')[0] , 64)
	# reg = 0.01* tf.nn.l2_loss(tf.trainable_variables(scope='dense/kernel:0')[0])

	# train_op = tf.train.AdamOptimizer(LR).minimize(tf.add(loss, reg))
	train_op = tf.train.AdamOptimizer(LR).minimize(loss)

	# print('train op defined')
	accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
	    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

	sess = tf.Session()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
	sess.run(init_op)     # initialize var in graph

	for step in range(num_epochs):    # training
		
		for batch in range(batch_size):		
			b_x, b_y, b_seq = next_batch(x_train, y_train, train_seq, batch, batch_size)
			# print('starting session to run')
			# print(len(b_seq))
			_, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y, seqlen:b_seq})
			# print('ending session to run')
			break

		preds = sess.run(output, {tf_x: x_test, seqlen:test_seq})
		print(multilabel(preds, y_test))

	print('------------ tensorflow final prediction accuracy ---------')
	preds = sess.run(output, {tf_x: x_test, seqlen:test_seq})
	print(plot_multilabel(preds, y_test))
	return preds

def next_batch(x,y, seq, batch, batch_size):
	i = batch*batch_size
	j = i + batch_size

	return x[i:j], y[i:j], seq[i:j]

def train_health(dense, dimension, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):

	look_ahead = 3
	timesteps = look_ahead-1
	input_dim = dimension
	
	#top = get_rare()
	print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + 'hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()

	if dense == False:
		input_dim = 284

	model.add(LSTM(hid_layer,input_shape=(timesteps, input_dim)))  # returns a sequence of vectors of dimension 32
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	# model.add(Dense(284, activation='softmax', kernel_regularizer=h_reg))
	# model.compile(loss='binary_crossentropy',
	#             optimizer='rmsprop')

	model.add(Dense(284, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	            optimizer='adam')

	# model.add(Dense(dimension, kernel_regularizer=h_reg ))
	# model.compile(loss=losses.mean_squared_error,
	#             optimizer='rmsprop')

	print("MODEL COMPILED. TRAINING AND VALIDATION STARTED.")

	model.fit(x_train, y_train,
	          batch_size=batch_size, epochs=num_epochs, verbose=2)

	return model

def nn(predict_test, y_test, embeddings):
	
	total_size = len(y_test)
	acc = 0.0
	score = 0.0

	for i in range(0,len(predict_test)):
		dist = np.zeros(284)
		for j in range(0,284):
			dist[j] = LA.norm(np.subtract(predict_test[i], embeddings[j]))
		
		top_pred = dist.argsort()[0:20]
		# if len(set(top_pred).intersection(set(y_test[i]))) >= 1:
		# 	score = score + 1

		acc = acc + float(len(set(top_pred).intersection(set(y_test[i]))))/float(len( set(y_test[i] )))
	
	# return float(score)/float(total_size)	
	return float(acc)/float(total_size)

def multilabel(predict_test, y_test):
	
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

		x_axis.append(test_seq[i])
		y_axis.append(curr_acc)

		# print((float(len(set(top_pred).intersection(y_test[i])))/float(len(y_test[i]))))
		# break

	# print('SCORE '+str(score))
	


	return float(acc)/float(total)

def plot_multilabel(predict_test, y_test):
	
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

		x_axis.append(test_seq[i])
		y_axis.append(curr_acc)

		# print((float(len(set(top_pred).intersection(y_test[i])))/float(len(y_test[i]))))
		# break

	# print('SCORE '+str(score))
	

	pyplot.scatter(x_axis,y_axis)
	pyplot.show()

	return float(acc)/float(total)

# dim_array = [2**x for x in range(3,8)]
# batch_size = [1024, 512, 256, 128]
# hid_layer = [32, 64, 128]
# dense = True
# num_epochs = 150
# pre = Preprocess()


# for dim in dim_array:
# 	x_train, y_train, x_test, y_test = pre.generate_data(dim, dense)
# 	for batch in batch_size:
# 		for hid in hid_layer:
# 			train_health(dense, dim, batch, hid, num_epochs, x_train, y_train, x_test, y_test)
dim = 32
dense = False
pre = Preprocess()

x_train, y_train, x_test, y_test, train_seq, test_seq = pre.generate_data(dim, dense)
preds = tf_train_health(dense, dim, 64, 64, 20, x_train, y_train, x_test, y_test, train_seq, test_seq)


# use DueTo relation to add more information into these sparse vectors
# print('Using DUE TO relation...')
# duef = open('dueTo.rel','rb')
# adjlist = pkl.load(duef)
# duef.close()

# for i in range(0,len(x_train)):
# 	for j in range(0,len(x_train[i])):
# 		for k in range(0,len(x_train[i][j])):
# 			if x_train[i][j][k] == 1:
# 				for ed in adjlist[k]:
# 					x_train[i][j][ed] = 1



# for i in range(0,len(x_test)):
# 	for j in range(0,len(x_test[i])):
# 		for k in range(0,len(x_test[i][j])):
# 			if x_test[i][j][k] == 1:
# 				for ed in adjlist[k]:
# 					x_test[i][j][ed] = 1



#train_health(dense, dimension, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):
#keras

# model = train_health(dense, dim, 64, 64, 5, x_train, y_train, x_test, y_test)
# predict_test = model.predict(x_test)
# print('keras accuracy')
# print(multilabel(predict_test,y_test))
# print(nn(predict_test,y_test,pre.embeddings))


#tensorflow
# preds = tf_train_health(dense, dim, 64, 64, 30, x_train, y_train, x_test, y_test)
# print(multilabel(preds, y_test))

#print(nn(model.predict(x_test),y_test,pre.embeddings))




# dim_array = [8,16,32,64,128]
# for dim in dim_array:
# 	x_train, y_train, x_test, y_test = pre.generate_data(dim, dense)
# 	model = train_health(dense, dim, 64, 128, 25, x_train, y_train, x_test, y_test)
# 	predict_test = model.predict(x_test)
# 	print('Dimension : '+str(dim))
# 	print(multilabel(predict_test,y_test))






