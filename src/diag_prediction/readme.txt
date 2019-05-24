READ_ME_FILE : 

The relevant files are :
main.py
embeddings.py
preprocess.py
mimic.py
proact.py


In main.py, you can change the following parameters : 
dataset = proact or mimic
lookahead : this parameter is used for number of timesteps we are considering. For eg., lookahead =2 indicates we are predicting t+1 timestep using t timestep.
dense : False => multihot encodings; True => dense encodings
dimension : dimension of the "dense=True" embeddings, otherwise, we use the number of classes as the dimension for multihot encodings
hidden layer : dimension of hidden layer in RNN
batch size: size of batch for training 
num_epochs : number of epochs used for training


The main workflow is as follows : 
1. Preparing the data. (preprocess.py, embeddings.py, mimic.py)
We haven't pickled the files for train, test. So, with default settings, a call is made to 
mimic.py to prepare necessary data. The relevant parameters here are (dense, dimension, lookahead).

When *dense=False*, the objects returned are train and test data. 
Train data contains examples for training and their labels.
x_train contains examples used for training and is a N x (lookahead-1) x 284 numpy array. 
y_train contains labels for corresponding examples used. It is a N x 284 numpy array.

Test data contains examples for testing and their labels.
x_test contains examples used for testing and is a M x (lookahead-1) x 284 numpy array. 
y_test contains labels for corresponding examples used. It is a M x _ numpy array. 
During training, our labels are encoded in a multi-hot way. Because, we used our own accuracy function, we defined test labels for convenience as follows. Each test label is basically a list of diagnosis entries for eg., [100, 200, 223].

When *dense=True*, we also give the dimension for the embeddings the objects returned are train and test data.
Train data contains examples for training and their labels.
x_train contains examples used for training and is a N x lookahead-1 x dimension numpy array. 
y_train contains labels for corresponding examples used. It is a N x 284 numpy array.

Test data contains examples for testing and their labels.
x_test contains examples used for testing and is a M x lookahead-1 x dimension numpy array. 
y_test contains labels for corresponding examples used. It is a M x _ numpy array. 
During training, our labels are encoded in a dense way. Because, we used our own accuracy function, we defined test labels for convenience as follows. Each test label is basically a list of diagnosis entries for eg., [100, 200, 223].

Some relevant pickled files in the repo : 
adm_dt : for every hospital admission, we extract date time object and pickle it
inv_idx : every icd9 code is mapped to a unique id between 0-283 based on classification.

2. Training (main.py)
After we have obtained the data, we use tensorflow for training a neural network with the following architecture

Input -> RNN -> Dense -> Output

For attention based model we have : 
Input -> Dense -> attention_weights

Input <dot> attention weights -> RNN -> Dense -> Output

3. Accuracy (main.py)
After obtaining the labels, we calculate accuracy using the following formulation : 
P : predicted labels
L : actual labels
acc = |P intersection L| / |L|


Methodology : 
We started with the goal of understanding how to apply machine learning techniques to health care datasets. We found MIMIC to be a diverse dataset containing information for about 50000 hospital visits in ICU. One of the classic problems in this domain is to understand how we can apply these techniques to predict the future diagnosis given patient history. We experimented quite a lot with 







