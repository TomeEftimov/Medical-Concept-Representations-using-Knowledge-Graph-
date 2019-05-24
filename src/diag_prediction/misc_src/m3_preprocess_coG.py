#preprocessing script for mimic3 dataset
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import os.path
import sys
from read_embeddings import embeddings

def co_occur(list_codes):
	for i in range(0,len(list_codes)):
		for j in range(i,len(list_codes)):
			x, y = list_codes[i],list_codes[j]
			co_G[x][y] = 1
			co_G[y][x] = 1

def generate_data_emb(data):
	
	embed_data = []

	for pt in data:
		pt_exp = []
		for i in range(0,len(pt)):
			pt_exp.append(embeddings[pt[i]])
	
		embed_data.append(pt_exp)
	
	embed_data = np.array(embed_data)
	
	return embed_data
	

def generate_data_one_hot(data):
	

	embed_data = []

	for pt in data:
		pt_exp = []
		for i in range(0,len(pt)):
			pt_exp.append(attr_dict[pt[i]])
	
		embed_data.append(pt_exp)
	
	embed_data = np.array(embed_data)
	
	return embed_data



pwd = './data/mimic3/'
file = pwd+'DIAGNOSES_ICD.csv'
look_ahead = 5
num_classes = 284

#initialize the co-occurrence graph
#co_G = [[0 for i in range(num_classes)] for j in range(num_classes)]

df = pd.read_csv(file)
td = df[['SUBJECT_ID','ICD9_CODE']]

data = []

fh = open('inv_idx','rb')
inv_idx = pkl.load(fh)
fh.close()

inv_idx['nan'] = 283

uless_count = 0
uful_count = 0
count_sub = 0
total_diag_count = 0
unique_icd9_codes = set([])
count_icd9_codes = {}

cap = 0

for name, group in td.groupby('SUBJECT_ID'):	

	cap = cap + 1
	

	count_sub = count_sub + 1
	total_diag_count = total_diag_count + len(group)

	if len(group) < look_ahead:
		uless_count = uless_count + 1
		continue

	uful_count = uful_count + 1

	icd9_codes = [inv_idx[str(x)] for x in np.array(group['ICD9_CODE'])]
	#co_occur(icd9_codes)

	#generating a sequence of lookahead points in a sliding fashion
	for i in range(0,len(icd9_codes)-look_ahead+1):

		outcome = 0
		pt = []

		for j in range(0,min(look_ahead,len(icd9_codes))) :
			# if code not in hierarchy:
			# 	print(icd9_codes[i+j,1])
			# 	sys.exit()
			# code = hierarchy[code]
			code = icd9_codes[i+j]
			
			if code in count_icd9_codes:
				count_icd9_codes[code] = count_icd9_codes[code] + 1
			else :
				count_icd9_codes[code] = 1

			pt.append(code)
			unique_icd9_codes = unique_icd9_codes | set(pt)

		data.append(pt)

print('done loading data')
print('useless data count '+str(uless_count))
print('useful data count '+str(uful_count))
print('avg icd9-count '+str(float(total_diag_count)/float(count_sub)))



# for i in range(0,(num_classes)):
# 	for j in range(0,(num_classes)):
# 		if co_G[i][j] == 1 and not i==j:
# 			print(str(i)+" "+str(j))

#generate one-hot encodings for mimic3 codes
attr_list = list(unique_icd9_codes)
attr_dict = {}

for i in range(0, len(attr_list)):
	v = [0]*len(attr_list)
	v[i] = 1
	attr_dict[attr_list[i]] = v

#generate train and test data
random.shuffle(data)

train_size = int(0.70*len(data))
data = np.array(data)
train = data[:train_size]
test = data[train_size+1:]

#train data
#train labels

x_train = []
y_train = []

train_d = train[:,:-1]
x_train = generate_data_emb(train_d)
#x_train = generate_data_one_hot(train_d)
train_l = train[:,-1]

#train labels
y_train = []
for tlabel in train_l:
	y_train.append(attr_dict[tlabel])

y_train = np.array(y_train)


test_d = test[:,:-1]
test_l = test[:,-1]
x_test = generate_data_emb(test_d)
#x_test = generate_data_one_hot(test_d)

#test labels
y_test = []
for tlabel in test_l:
     y_test.append(attr_dict[tlabel])

y_test = np.array(y_test)