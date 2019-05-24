#preprocessing script for mimic2 dataset
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import os.path

pwd = './data/32/'
lookahead = 5

data = []


uless_count = 0
uful_count = 0
total_num_entries = 0
unique_icd9_codes = set([])
count_icd9_codes = {}

for patient in range(32000,33001):
	folder = str(patient)
	file = 'ICD9-'+str(patient)+'.txt'

	path_f = pwd +folder+'/'+file

	#check if the file exists
	if os.path.isfile(path_f)==False:
		continue

	num_lines = sum(1 for line in open(path_f))
	if num_lines >= 6:
		uful_count = uful_count + 1
	else:
		uless_count = uless_count + 1

	total_num_entries = total_num_entries + num_lines-1 #first line contains metadata

	df = pd.read_csv(path_f)
	td = df['CODE']
	icd9_codes = td.values
	

	#iterate over the codes and collapse the hierarchy
	icd9_codes = [str(c) for c in icd9_codes]
	icd9_codes = [c.split('.')[0] for c in icd9_codes]

	#identify the frequency of the codes
	for c in icd9_codes:
		if c in count_icd9_codes:
			count_icd9_codes[c] = count_icd9_codes[c] + 1
		else :
			count_icd9_codes[c] = 1

	#total number of unique_icd9_codes
	unique_icd9_codes = unique_icd9_codes | set(icd9_codes)

	# assume that icd9_codes has length atleast size of lookahead, else it will not enter this for-loop
	for i in range(0, len(icd9_codes)-lookahead+1):
		pt = []
		for j in range(0,lookahead):
			pt.append(icd9_codes[i+j])

		data.append(pt)	

print('done loading data')
print('useless data count '+str(uless_count))
print('useful data count '+str(uful_count))
print('avg icd9-count '+str(float(total_num_entries)/(uless_count+uful_count)))

#pickling metadata
mimic2 = "_mimic2"
f_count = open("count_"+mimic2, "wb")
pkl.dump(count_icd9_codes, f_count)
f_count.close()

f_unique = open("unique_icd9_codes"+mimic2, "wb")
pkl.dump(unique_icd9_codes, f_unique)
f_unique.close()

#generate one-hot encodings for mimic2 codes
attr_list = list(unique_icd9_codes)
attr_dict = {}
inv_idx = {}
for i in range(0, len(attr_list)):
     v = [0]*len(attr_list)
     v[i] = 1
     attr_dict[attr_list[i]] = v
     inv_idx[i] = attr_list[i]

f_code_dict = open("code_dict"+mimic2, "wb")
pkl.dump(inv_idx, f_code_dict)
f_code_dict.close()

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
train_l = train[:,-1]  
train_d = train[:,:-1]

for tra in train_d:
     tr_exp = []
     for i in range(0,len(tra)):
          tr_exp.append(attr_dict[tra[i]])
     x_train.append(tr_exp)

x_train = np.array(x_train)

#train labels
for tlabel in train_l:
	y_train.append(attr_dict[tlabel])

y_train = np.array(y_train)

#test labels
#test data

test_l = test[:,-1]  
test_d = test[:,:-1]

x_test = []
y_test = []

for tes in test_d:
     ts_exp = []
     for i in range(0,len(tes)):
          ts_exp.append(attr_dict[tes[i]])
     x_test.append(ts_exp)

x_test = np.array(x_test)

#test labels
for tlabel in test_l:
     y_test.append(attr_dict[tlabel])

y_test = np.array(y_test)

#have to pickle the data

f_xtr = open("x_train"+mimic2, "wb")
pkl.dump(x_train, f_xtr)
f_xtr.close()

f_ytr = open("y_train"+mimic2, "wb")
pkl.dump(y_train, f_ytr)
f_ytr.close()

f_xte = open("x_test"+mimic2, "wb")
pkl.dump(x_test, f_xte)
f_xte.close()

f_yte = open("y_test"+mimic2, "wb")
pkl.dump(y_test, f_yte)
f_yte.close()