from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.layers import Activation, GRU, LSTM, Dense, Dropout
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import tensorflow as tf 
#from mimic3_preprocess import unique_icd9_codes, count_icd9_codes, inv_idx, x_train, y_train, x_test, y_test
#from m3_preprocess_coG import unique_icd9_codes, count_icd9_codes, inv_idx, x_train, y_train, x_test, y_test
from preprocess_new import Preprocess

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

	

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 5) 


def train_health(dense, dimension, batch_size, hid_layer,  num_epochs):

	look_ahead = 5
	timesteps = look_ahead-1
# mimic2 = "_mimic2"

# f_count = open("count_"+mimic2, "rb")
# count_icd9_codes = pkl.load(f_count)
# f_count.close()

# f_unique = open("unique_icd9_codes"+mimic2, "rb")
# unique_icd9_codes = pkl.load( f_unique)
# f_unique.close()

#data_dim = len(unique_icd9_codes) #number of attributes/icd9-codes
#data_dim = len(x_train[0][0])

# f_code_dict = open("code_dict"+mimic2, "rb")
# code_dict = pkl.load( f_code_dict)
# f_code_dict.close()
#code_dict = inv_idx

# f_xtr = open("x_train"+mimic2, "rb")
# x_train = pkl.load(f_xtr)
# f_xtr.close()


# f_ytr = open("y_train"+mimic2, "rb")
# y_train = pkl.load(f_ytr)
# f_ytr.close()


# f_xte = open("x_test"+mimic2, "rb")
# x_test = pkl.load(f_xte)
# f_xte.close()


# f_yte = open("y_test"+mimic2, "rb")
# y_test = pkl.load(f_yte)
# f_yte.close()


#top = get_rare()

hid_layer = 64
num_epochs = 500
batch_size = 256
data_dim = 8

print('hidden layer : '+str(hid_layer)+' num of epochs : '+str(num_epochs)+' batch size : '+str(batch_size))

# expected input data shape: (batch_size, timesteps, data_dim)

model = Sequential()
#model.add(GRU(hid_layer,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(Activation('relu'))

model.add(GRU(hid_layer,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(Activation('relu'))
model.add(Dropout(0.2))

#model.add(LSTM(hid_layer, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(hid_layer, return_sequences=True))   # returns a sequence of vectors of dimension 32
#model.add(LSTM(hid_layer))  # return a single vector of dimension 32

model.add(Dense(282, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy',top_5_accuracy])

# Generate dummy training data
#x_train = np.random.random((1000, timesteps, data_dim))
#y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
#x_val = np.random.random((100, timesteps, data_dim))
#y_val = np.random.random((100, num_classes))
pre = Preprocess()
x_train, y_train, x_test, y_test = pre.generate_data(data_dim, True)

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test))

#normal_rv = top_k_categorical_accuracy(y_test,model.predict(x_test),k=5)

#initialize the variable
#init_op = tf.initialize_all_variables()

#run the graph
# with tf.Session() as sess:
#     sess.run(init_op) #execute init_op
#     #print the random values that we sample
#     print (sess.run(normal_rv))

# y_pred = model.predict(x_test)
# acc = top_k_categorical_accuracy(y_test, y_pred, k=5)
# print(acc)

#print(accuracy_5(y_test,model.predict(x_test)))

#grid_search : 
dim_array = [2**x for x in range(2,8)]
batch_size = [64, 128, 256, 512, 1024]
hid_layer = [32, 64, 128]
dense_array = [True, False]
num_epochs = 10

for dense in dense_array: 
	for dim in dim_array:
		for batch in batch_size:
			for hid in hid_layer:
				train_health(dense, dim, batch, hid, num_epochs)










