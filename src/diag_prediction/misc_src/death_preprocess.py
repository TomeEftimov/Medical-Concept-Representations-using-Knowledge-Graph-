#preprocessing script for mimic3 dataset
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import os.path
import sys
from read_embeddings import GraphEmbeddings


class Preprocess():

	data = []
	train = []
	test = []
	vertexEmb = None
	embeddings = {}
	attr_dict = {}
	attr_list = []
	adm_dt = {}
	adm_death = {}
	inv_idx = {}

	def co_occur(self,list_codes):
		for i in range(0,len(list_codes)):
			for j in range(i,len(list_codes)):
				x, y = list_codes[i],list_codes[j]
				co_G[x][y] = 1
				co_G[y][x] = 1

	def generate_dense_enc(self,data, dimension, embeddings):
		
		embed_data = []

		
		for pt in data:
			
			#print()
			pt_exp = []
			
			for i in range(0,len(pt)):
				
				v = np.zeros(dimension)
				
				for j in range(0,len(pt[i])):
					#print(len(pt[i]))
					#print(embeddings[pt[i][j]])
					v = np.add(v, np.array( embeddings[pt[i][j]]   ))
					#v = np.sum(v, np.array(embeddings[ pt[i][j] ]))

				v = v/len(pt[i])
				pt_exp.append(v)

			embed_data.append(pt_exp)

		return np.array(embed_data)


		# for pt in data:
		# 	pt_exp = []
		# 	for i in range(0,len(pt)):
		# 		pt_exp.append(embeddings[pt[i]])
		
		# 	embed_data.append(pt_exp)
		
		# embed_data = np.array(embed_data)
		
		# return embed_data
		

	def generate_onehot_enc(self,data):		
		embed_data = []

		for pt in data:
			# pt_exp = []
			# for i in range(0,len(pt)):
			# 	pt_exp.append(attr_dict[pt[i]])
		
			# embed_data.append(pt_exp)
			seq = []
			pt_exp = [0]*284
			for i in range(0,len(pt)):
				for j in range(0,len(pt[i])):
					pt_exp[pt[i][j]]= 1
				
				seq.append(pt_exp)

			embed_data.append(seq)

		embed_data = np.array(embed_data)
		
		return embed_data

	def generate_encodings(self,dimension, dense):
		
		x_train = []
		y_train = []

		train_d = self.train[:,:-1]
		train_l = self.train[:,-1]
		test_d = self.test[:,:-1]
		test_l = self.test[:,-1]

		#labels
		y_train = []
		y_test = []

		if dense == False: #generate one-hot encodings
			x_train = self.generate_onehot_enc(train_d)
			x_test = self.generate_onehot_enc(test_d)

		else:
			self.embeddings = self.vertexEmb.getEmbeddings(dimension)
			x_train = self.generate_dense_enc(train_d, dimension, self.embeddings)
			x_test = self.generate_dense_enc(test_d, dimension, self.embeddings)
			
		# for tlabel in train_l:
		# 	v = np.zeros(dimension)
		# 	for t in tlabel:
		# 		v = np.add(v, self.embeddings[t])

		# 	v = v/len(tlabel)	
		# 	y_train.append(v)

		# multi label classification approach
		# for tlabel in train_l:
		# 	v = np.zeros(284)
		# 	for t in tlabel:
		# 		v[t] = 1

		# 	y_train.append(v)
		y_train = []
		for tl in train_l:
			if tl == 0:
				y_train.append(np.array([0,1]))
			if tl == 1:
				y_train.append(np.array([1,0]))

		y_train = np.array(y_train)

		y_test = []
		for tel in test_l:
			if tel == 0:
				y_test.append(np.array([0,1]))
			if tel == 1:
				y_test.append(np.array([1,0]))


		#test labels
		# for tlabel in test_l:
		# 	v = np.zeros(dimension)
			
		# 	for t in tlabel:
		# 		v = np.add(v, embeddings[t])
			
		# 	v = v/len(tlabel) 
		# 	y_test.append(v)
		
		# for tlabel in test_l:
		# 	v = np.zeros(284)
		# 	for t in tlabel:
		# 		v[t] = 1

		# 	y_test.append(v)

		y_test = np.array(y_test)

		return x_train, y_train, x_test, y_test

	def generate_data(self,dimension, dense):

		for i in range(0, len(self.attr_list)):
			v = [0]*len(self.attr_list)
			v[i] = 1
			self.attr_dict[self.attr_list[i]] = v

		#generate train and test data
		random.shuffle(self.data)

		train_size = int(0.70*len(self.data))
		self.data = np.array(self.data)
		self.train = self.data[:train_size]
		self.test = self.data[train_size+1:]

		return self.generate_encodings(dimension, dense)

	def cmp_to_key(self,mycmp):
	    
	    'Convert a cmp= function into a key= function'

	    class K(object):
	    	def __init__(self, obj, *args):
	    		self.obj = obj

	    	def __lt__(self, other):
	    		return mycmp(self.obj, other.obj) < 0

	    	def __gt__(self, other):
	    		return mycmp(self.obj, other.obj) > 0

	    	def __eq__(self, other):
	    		return mycmp(self.obj, other.obj) == 0

	    	def __le__(self, other):
	    		return mycmp(self.obj, other.obj) <= 0

	    	def __ge__(self, other):
	    		return mycmp(self.obj, other.obj) >= 0

	    	def __ne__(self, other):
	    		return mycmp(self.obj, other.obj) != 0

	    return K

	def compare_datetime(self,h1, h2):

		# print(h1,type(h1))

		date1, time1 = self.adm_dt[h1].split(' ')
		year1, month1, day1 = map(int,date1.split('-'))
		hour1, minutes1, sec1 = map(int,time1.split(':'))

		date2, time2 = self.adm_dt[h2].split(' ')
		year2, month2, day2 = map(int,date2.split('-'))
		hour2, minutes2, sec2 = map(int,time2.split(':'))
	    
		if year1 > year2:
		    return 1
		elif year1 < year2:
		    return -1
		elif month1 > month2:
		    return 1
		elif month1 < month2:
		    return -1
		elif day1 > day2:
		    return 1
		elif day1 < day2:
		    return -1
		elif hour1 > hour2:
		    return -1
		elif hour1 < hour2:
		    return -1
		elif minutes1 > minutes2:
		    return 1
		elif minutes1 < minutes2:
		    return -1
		elif sec1 >= sec2:
		    return 1
		else:
		    return 0

	def preprocess(self):
		pwd = './data/mimic3/'
		file = pwd+'DIAGNOSES_ICD.csv'
		

		#initialize the co-occurrence graph
		#co_G = [[0 for i in range(num_classes)] for j in range(num_classes)]

		df = pd.read_csv(file)
		td = df[['SUBJECT_ID','HADM_ID','ICD9_CODE']]

		look_ahead = 5
		num_classes = 284

		self.inv_idx['nan'] = 283

		uless_count = 0
		uful_count = 0
		count_sub = 0
		total_diag_count = 0
		unique_icd9_codes = set([])
		count_icd9_codes = {}
		
		self.data = []
		cd = 0
		cl = 0
		for name, group in td.groupby('SUBJECT_ID'):	

			l = group.groupby(['HADM_ID'])

			if len(l) >= 2:
				
				sub = group['SUBJECT_ID'].values[0]
				hadm_code_dict = {}
				hadm_list = []
				for c in l:
					hadm_list.append(c[0])
					icd9_codes = [self.inv_idx[str(x)] for x in c[1]['ICD9_CODE'].values]
					hadm_code_dict[c[0]] = icd9_codes

					# print(hadm_list)	

				hadm_list.sort(key=self.cmp_to_key(self.compare_datetime))
				
				for i in range(0,len(hadm_list)-1):
					status = 0
					if str(self.adm_death[hadm_list[i+1]]) == 'nan':
						status= 0
						cd = cd + 1 
						if cd <= 10000:
							self.data.append([hadm_code_dict[hadm_list[i]], hadm_code_dict[hadm_list[i+1]], status ])

					else:
						status = 1
						cl = cl + 1
						if cl <= 10000:
							self.data.append([hadm_code_dict[hadm_list[i]], hadm_code_dict[hadm_list[i+1]], status ])

					# self.data.append([hadm_code_dict[hadm_list[i]], hadm_code_dict[hadm_list[i+1]], status ])

		self.attr_list = list(unique_icd9_codes)
		self.attr_dict = {}


	def __init__(self):
		self.vertexEmb = GraphEmbeddings()
		 


		self.data = []

		fh = open('inv_idx','rb')
		self.inv_idx = pkl.load(fh)
		fh.close()

		fdt = open('adm_dt','rb')
		self.adm_dt = pkl.load(fdt)
		fdt.close()

		fdth = open('adm_death','rb')
		self.adm_death = pkl.load(fdth)
		fdth.close()

		self.preprocess()

		
	