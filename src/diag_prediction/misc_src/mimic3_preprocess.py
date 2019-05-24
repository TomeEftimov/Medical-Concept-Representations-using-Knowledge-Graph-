#preprocessing script for mimic3 dataset
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import os.path
import sys

def get_ancestor(code):

	anc = code

	if anc[0]=='V':
		anc = anc[0]+anc[1]+anc[2]
	elif anc[0] == 'E':
		anc = anc[0]+anc[1]+anc[2]+anc[3]
	else:
		anc = anc[0]+anc[1]+anc[2]

	return anc


def co_occur(i, j):
	G[i,j] = 1

pwd = './data/mimic3/'
file = pwd+'DIAGNOSES_ICD.csv'
look_ahead = 5
G = [[0 for i in range(283)] for j in range(283)]

df = pd.read_csv(file)
td = df[['SUBJECT_ID','ICD9_CODE']]

data = []

fh = open('hierarchy','rb')
hierarchy = pkl.load(fh)
fh.close()
hierarchy['nan'] = '20'

uless_count = 0
uful_count = 0
count_sub = 0
total_diag_count = 0
unique_icd9_codes = set([])
count_icd9_codes = {}

cap = 0

for name, group in td.groupby('SUBJECT_ID'):

	

	count_sub = count_sub + 1
	total_diag_count = total_diag_count + len(group)

	if len(group) < look_ahead:
		uless_count = uless_count + 1
		continue

	uful_count = uful_count + 1

	icd9_codes = group.values
	for i in range(0,len(icd9_codes)-look_ahead+1):

		outcome = 0
		pt = []

		for j in range(0,min(look_ahead,len(icd9_codes))) :

			code = str(icd9_codes[i+j,1]).split('.')[0]
			if code not in hierarchy:
				print(icd9_codes[i+j,1])
				sys.exit()

			code = hierarchy[code]
			
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

#pickling metadata
# mimic3 = "_mimic3"
# f_count = open("count_"+mimic3, "wb")
# pkl.dump(count_icd9_codes, f_count)
# f_count.close()

# f_unique = open("unique_icd9_codes"+mimic3, "wb")
# pkl.dump(unique_icd9_codes, f_unique)
# f_unique.close()

#generate one-hot encodings for mimic3 codes
attr_list = list(unique_icd9_codes)
attr_dict = {}
inv_idx = {}
for i in range(0, len(attr_list)):
     v = [0]*len(attr_list)
     v[i] = 1
     attr_dict[attr_list[i]] = v
     inv_idx[i] = attr_list[i]

# f_code_dict = open("code_dict"+mimic3, "wb")
# pkl.dump(inv_idx, f_code_dict)
# f_code_dict.close()

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

# f_xtr = open("x_train"+mimic3, "wb")
# pkl.dump(x_train, f_xtr)
# f_xtr.close()

# f_ytr = open("y_train"+mimic3, "wb")
# pkl.dump(y_train, f_ytr)
# f_ytr.close()

# f_xte = open("x_test"+mimic3, "wb")
# pkl.dump(x_test, f_xte)
# f_xte.close()

# f_yte = open("y_test"+mimic3, "wb")
# pkl.dump(y_test, f_yte)
# f_yte.close()