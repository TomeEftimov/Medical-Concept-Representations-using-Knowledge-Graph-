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
	
	l = []
	mean = 0.
	std = 0.
	
	x_train = []
	x_test = []
	y_train = []
	y_test  = []

	num_classes = 0

	def co_occur(self,list_codes):
		for i in range(0,len(list_codes)):
			for j in range(i,len(list_codes)):
				x, y = list_codes[i],list_codes[j]
				self.co_G[x][y] += 1
				self.co_G[y][x] += 1

	def generate_onehot_enc(self,data):	
		
		embed_data = []

		for point in data:
			
			seq = []
			for i in range(0,len(point)):
				
				pt_exp = [0]*284
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
				v[t] = 1

			y_train.append(v)

		y_train = np.array(y_train)


		y_test = np.array(test_l)

		return x_train, y_train, x_test, y_test

	def generate_data(self,data, dense, dimension):

		#generate train and test data
		# random.shuffle(data)

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
		# num_vertices = 6985
		num_vertices = num_classes
		self.co_G = [[0 for i in range(num_vertices)] for j in range(num_vertices)]

		df = pd.read_csv(file)
		td = df[['SUBJECT_ID','HADM_ID','ICD9_CODE']]


		fh = open('inv_idx','rb')
		inv_idx = pkl.load(fh)
		fh.close()

		fh = open('code_dict','rb')
		code_dict = pkl.load(fh)
		fh.close()
		code_dict['nan'] = 0

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

		data = []
		for name, group in td.groupby('SUBJECT_ID'):	

			l = group.groupby(['HADM_ID'])				
			sub = group['SUBJECT_ID'].values[0]
			hadm_code_dict = {}
			hadm_list = []

			for c in l:
				hadm_list.append(c[0])
				icd9_codes = [inv_idx[str(x)] for x in c[1]['ICD9_CODE'].values]
				self.co_occur(icd9_codes)

		
		l = []
		graph = self.co_G

		for i in range(0, len(graph)):
			for j in range(i+1, len(graph[i])):
				if graph[i][j] >= 1 and (not (i == j)):
					l.append(graph[i][j])

		self.l = np.array(l)
		self.mean = np.mean(l)
		self.std = np.std(l)

	def get_graph(self,fname='', boundary=0):

		fg = open(fname,'w')
		num_ed = 0
		graph = self.co_G
		neighbors = {}
		for i in range(0, len(graph)):
			# neighbors[i] = np.array(graph[i]).argsort()[-5:]
			for j in range(i+1, len(graph[i])):
				if graph[i][j] >= boundary and (not (i == j)):
					fg.write(str(i)+' '+str(j)+'\n')
					num_ed += 1
		fg.close() 
		print('number of edges ',str(num_ed))

		return neighbors

m = mimic(lookahead=3,dense=False, dimension=32)
neighbors = m.get_graph('graph_full.txt')