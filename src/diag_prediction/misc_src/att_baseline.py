from keras.metrics import top_k_categorical_accuracy
from keras import losses
from keras.models import Sequential, Model
from keras.layers import Activation, GRU, LSTM, Dense, Dropout, Input, Permute, Reshape, Multiply
from keras import regularizers
from keras import backend as K
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import tensorflow as tf 
from numpy import linalg as LA


#from mimic3_preprocess import unique_icd9_codes, count_icd9_codes, inv_idx, x_train, y_train, x_test, y_test
#from m3_preprocess_coG import unique_icd9_codes, count_icd9_codes, inv_idx, x_train, y_train, x_test, y_test
from hadm_preprocess import Preprocess
#from hadm_preprocess import PreprocessMulti

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
	
	# Documentation is available online on Github at the address below.
	# From: https://github.com/philipperemy/keras-visualize-activations
	
	print('----- activations -----')
	activations = []
	inp = model.input
	if layer_name is None:
		outputs = [layer.output for layer in model.layers]
	else:
		outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
	funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
	layer_outputs = [func([inputs, 1.])[0] for func in funcs]
	for layer_activations in layer_outputs:
		activations.append(layer_activations)
		if print_shape_only:
			print(layer_activations.shape)
		else:
			print(layer_activations)
	return activations

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

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 5)


#attention applied before the lstm layer. 
def attention(inputs, TIME_STEPS=2):

	input_dim = int(inputs.shape[2])
	# inputs.shape = (batch_size, time_steps, input_dim)
    #input_dim = int(inputs.shape[2])
	a = Permute((2, 1))(inputs)
	#a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
	a = Dense(TIME_STEPS, activation='softmax')(a)

	a_probs = Permute((2, 1), name='attention_vec')(a)
	output_attention_mul = Multiply()([inputs, a_probs])
	return output_attention_mul

def train_health(dense, dimension, batch_size, hid_layer,  num_epochs, x_train, y_train, x_test, y_test):

	look_ahead = 2
	timesteps = look_ahead-1
	#top = get_rare()

	print('dense : '+str(dense) + ' dimension : '+str(dimension)+' batch size : '+str(batch_size) + 'hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs))

	# expected input data shape: (batch_size, timesteps, data_dim)
	inputs = Input(shape=(timesteps, 284,))
	model = attention(inputs)
	model = GRU(hid_layer, input_shape=(timesteps, 284), activation='relu', dropout=0.5)(model)
	
	#model = Sequential()
	#model.add(GRU(hid_layer,input_shape=(timesteps, dimension)))  # returns a sequence of vectors of dimension 32
	#model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	#model.add(Dense(284, activation='softmax', kernel_regularizer=h_reg))
	
	output = Dense(284, activation='softmax',kernel_regularizer=h_reg)(model)
	model = Model(input=[inputs],output= output)
	# model.add(Dense(dimension, kernel_regularizer=h_reg ))
	# model.compile(loss=losses.mean_squared_error,
	#             optimizer='rmsprop')

	print("MODEL COMPILED. TRAINING AND VALIDATION STARTED.")
	model.compile(loss='binary_crossentropy',
		            optimizer='adam')

	print(model.summary())
	model.fit([x_train], y_train,
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
	total = len(predict_test)

	for i in range(0,total):
#		predict_test[i] vs y_test[i]
		preds = predict_test[i]
		top_pred = preds.argsort()[-10:]
		
		# trueL = []
		
		# for k in top_pred:
		# 	if trueL[k] == 1:	
		# 		score = score + 1
		# 		break
		acc = acc + (float(len(set(top_pred).intersection(set(y_test[i]) )))/float(len(set(y_test[i]))))
		# print((float(len(set(top_pred).intersection(y_test[i])))/float(len(y_test[i]))))
		# break

	# print('SCORE '+str(score))
	return float(acc)/float(total)



if __name__ == '__main__':

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
	dim = 8
	dense = False
	pre = Preprocess()
	x_train, y_train, x_test, y_test = pre.generate_data(dim, dense)
	model = train_health(True, dim, 64, 32, 5, x_train, y_train, x_test, y_test)

	attention_vectors = []
	attention_vector = np.mean(get_activations(model, 
													x_test, 
													print_shape_only=True,
													layer_name='attention_vec')[0], axis=2).squeeze()
	print('attention =', attention_vector)
	assert (np.sum(attention_vector) - 1.0) < 1e-5
	attention_vectors.append(attention_vector)

	attention_vector_final = np.mean(np.array(attention_vectors), axis=0)


	predict_test = model.predict([x_test])
	print(len(predict_test))
	#print(nn(predict_test,y_test,pre.embeddings))
	#print(multilabel(model.predict(x_test),y_test))









