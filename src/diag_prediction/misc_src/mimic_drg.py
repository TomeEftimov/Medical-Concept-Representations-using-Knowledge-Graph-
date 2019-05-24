import pandas as pd  
import pickle as pkl
import numpy as np
import random
import os.path
import sys

from checkin_embeddings import GraphEmbeddings

class mimic():

	# attr_list = []
	adm_dt = {}
	co_G = []
	vertexEmb = None

	x_train = []
	x_test = []
	y_train = []
	y_test  = []

	num_classes = 0
	drg_dict  = {}

	def __init_drugs(self):

		drg_dict = {}
		drg_list = set([])


		with open('hadm_drg.pkl','rb') as f:
			hadm_drg = pkl.load(f)			
			
			for adm in hadm_drg.keys():
				drg_list = set(hadm_drg[adm]) | drg_list 

			drg_list = list(drg_list)
			for i in range(0, len(drg_list)):
				drg_dict[ drg_list[i] ] = i+284   #simple hack by adding 284, we ensure all drug codes are above 284

		return hadm_drg, drg_dict

	def __adm_drugs(self, adm, hadm_drg, drg_dict):

		l = []
		
		if adm not in hadm_drg.keys():
			return l

		for drg in hadm_drg[adm]:
			if drg not in drg_dict.keys():
				print('>>>>>>>>>>>>>>>>>>   ', adm, drg)
			else:
				l.append(drg_dict[drg]) 
		return l


	def co_occur(self,list_codes):
		for i in range(0,len(list_codes)):
			for j in range(i,len(list_codes)):
				x, y = list_codes[i],list_codes[j]
				co_G[x][y] = 1
				co_G[y][x] = 1

	def generate_onehot_enc(self,data):	
		
		embed_data = []

		for point in data:
			
			seq = []
			for i in range(0,len(point)):
				
				pt_exp = [0]*(284+1667)
				for j in range(0,len(point[i])):
					pt_exp[point[i][j]]= 1

				seq.append(pt_exp)

			embed_data.append(seq)
		
		return np.array(embed_data)

	def generate_dense_enc(self, data, dimension ):
		
		embed_data = []
		embeddings = self.embeddings

		for pt in data:
			pt_exp = []
			for i in range(0,len(pt)):
				
				v = np.zeros(dimension)
				
				for j in range(0,len(pt[i])):
					v = np.add(v, np.array( embeddings[pt[i][j]] ))

				v = v/len(pt[i])
				pt_exp.append(v)

			embed_data.append(pt_exp)

		return np.array(embed_data)


	def generate_encodings(self,train, test, dense, dimension):
		
		x_train = []
		x_test = []

		train_d = train[:,:-1]
		train_l = train[:,-1]
		test_d = test[:,:-1]
		test_l = test[:,-1]

		#labels
		y_train = []
		y_test = []

		if dense == False: #generate one-hot encodings
			x_train = self.generate_onehot_enc(train_d)
			x_test = self.generate_onehot_enc(test_d)

		else:
			self.embeddings = self.vertexEmb.getEmbeddings(dimension)
			x_train = self.generate_dense_enc(train_d, dimension)
			x_test = self.generate_dense_enc(test_d, dimension)
			

		# multi label classification approach
		for tlabel in train_l:
			v = np.zeros(284)
			for t in tlabel:
				if t < 284:
					v[t] = 1

			y_train.append(v)

		y_train = np.array(y_train)

		labels = []
		for tlabel in test_l:
			v = []
			for t in tlabel:
				if t < 284:
					v.append(t)

			labels.append(v)

		y_test = np.array(labels)

		return x_train, y_train, x_test, y_test

	def generate_data(self,data, dense, dimension):

		#generate train and test data
		random.shuffle(data)

		train_size = int(0.80*len(data))
		data = np.array(data)
		train = data[:train_size]
		test = data[train_size+1:]

		return self.generate_encodings(train, test, dense, dimension)

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

	def __init__(self, lookahead, dense, dimension):
		self.mimic_prep(lookahead, dense, dimension)

	def mimic_prep(self, lookahead, dense, dimension):
		self.vertexEmb = GraphEmbeddings()
		 
		pwd = './data/mimic3/'
		file = pwd+'DIAGNOSES_ICD.csv'

		num_classes = 284

		#initialize the co-occurrence graph
		self.co_G = [[0 for i in range(num_classes)] for j in range(num_classes)]

		df = pd.read_csv(file)
		td = df[['SUBJECT_ID','HADM_ID','ICD9_CODE']]


		fh = open('inv_idx','rb')
		inv_idx = pkl.load(fh)
		fh.close()

		fdt = open('adm_dt','rb')
		self.adm_dt = pkl.load(fdt)
		fdt.close()

		inv_idx['nan'] = 283

		uless_count = 0
		uful_count = 0
		count_sub = 0
		total_diag_count = 0
		unique_icd9_codes = set([])
		count_icd9_codes = {}

		hadm_drg, drg_dict = self.__init_drugs()
		self.drg_dict = drg_dict

		data = []
		for name, group in td.groupby('SUBJECT_ID'):	

			l = group.groupby(['HADM_ID'])

			if len(l) >= lookahead:
				
				sub = group['SUBJECT_ID'].values[0]
				hadm_code_dict = {}
				hadm_list = []

				for c in l:
					hadm_list.append(c[0])
					icd9_codes = [inv_idx[str(x)] for x in c[1]['ICD9_CODE'].values]
					hadm_code_dict[c[0]] = icd9_codes

				hadm_list.sort(key=self.cmp_to_key(self.compare_datetime))

				for i in range(0,len(hadm_list)-lookahead+1):
					
					seq = []
					for j in range(0, lookahead):
						seq.append(hadm_code_dict[hadm_list[i+j]] + self.__adm_drugs(hadm_list[i+j], hadm_drg, drg_dict))

					data.append(seq)

		# self.attr_list = list(unique_icd9_codes)
		self.num_classes = 284
		self.x_train, self.y_train, self.x_test, self.y_test = self.generate_data(data, dense, dimension)
		# print('-------------- Data is Ready!! --------------')