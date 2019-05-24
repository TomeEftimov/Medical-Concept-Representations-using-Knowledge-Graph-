import pickle as pkl
import numpy as np

path_to_file = 'data/mimic3/MultiDX.txt'

#hierarchy = {}
inv_idx = {}
inv_map = {}

with open(path_to_file,'r') as f:
	data = f.read()

data = data.split('\n')

inv_name = {}

var_idx_1 = 0
var_idx_2 = 0
li = []
iidx = -1

for idx in range(0,len(data)):
	var_idx_2 = idx
	value = []
	key = ""
	if data[var_idx_2] == "":

		iidx = iidx + 1
		key = data[var_idx_1].split(" ")[0]
		inv_name[iidx] = ' '.join(data[var_idx_1].split(" ")[1:])

		inv_map[iidx] = key

		#extract the data between var_idx 1 and 2
		for i in range(var_idx_1+1, var_idx_2):
			l = [x for x in data[i].split(" ") if not x == '']
			
			# adding inverse lookup between the icd-code and collapsed heirarchy
			for x in l : 
				inv_idx[x] = iidx

			value = value + l
		
		#hierarchy[key] = value

		#set the new var_idx
		var_idx_1 = var_idx_2 + 1

#now pickle the hierarchy
# fpkl = open('inv_idx','wb')
# pkl.dump(inv_idx, fpkl)
# fpkl.close()
