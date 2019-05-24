import pandas as pd  
import pickle as pkl
import numpy as np
import random

class proact():

	soc_dict = {}
	inv_soc_dict = {}

	num_classes = 0

	x_train = []
	y_train = []
	x_test = []
	y_test = []

	def generate_data(self, data, dense, dimension):

		new_data = []
		for pt in data:
			seq  = []
			for ex in pt:
				seq.append([self.soc_dict[item] for item in ex])

			new_data.append(seq)

		data = new_data
		random.shuffle(data)

		train_size = int(0.80*len(data))
		data = np.array(data)

		train = data[:train_size]
		test = data[train_size+1:]

		return self.generate_encodings(train, test, dense, dimension)

	def generate_encodings(self, train, test, dense, dimension):

		x_train = []
		y_train = []

		x_test = []
		y_test = []


		train_d = train[:,:-1]
		train_l = train[:,-1]

		test_d = test[:,:-1]
		test_l = test[:,-1]
		
		if dense == False:
			x_train = self.generate_onehot_enc(train_d)
			x_test = self.generate_onehot_enc(test_d)

		#multilabel classification
		for tlabel in train_l:
			v = np.zeros(self.num_classes)
			for t in tlabel:
				v[t] = 1

			y_train.append(v)

		y_train = np.array(y_train)

		y_test = np.array(test_l)

		return x_train, y_train, x_test, y_test


	def generate_onehot_enc(self,data):	
		
		embed_data = []

		for point in data:
			
			seq = []
			for i in range(0,len(point)):
				
				pt_exp = [0]*self.num_classes
				for j in range(0,len(point[i])):
					pt_exp[point[i][j]]= 1

				seq.append(pt_exp)

			embed_data.append(seq)
		
		return np.array(embed_data)

	def __init__(self, lookahead = 3, dense= False, dimension= 8):

		data = []

		df = pd.read_csv('data/proact/AdverseEvents.csv')
		df.loc[df['Outcome'] != 'Death','Outcome'] = 0
		df.loc[df['Outcome'] == 'Death','Outcome'] = 1
		df.loc[df['Lowest_Level_Term'] == 'Death','Outcome'] = 1

		td = df[['subject_id','SOC_Abbreviation','Outcome', 'SOC_Code', 'Start_Date_Delta','End_Date_Delta']]
		#td = pd.read_pickle('td.pkl')

		x_train = []
		y_train = []

		soc_idx = 1
		out_idx = 2
		code_idx= 3
		start_time = 4
		end_time = 5

		idx = code_idx
		
		soc_abbr_list = set([])

		for name, group in td.groupby('subject_id'):

			sf = group.values

			if len(sf) < lookahead :
				continue

			timed_events = []
			curr_time = sf[0, 4]

			curr_event = []
			for i in range(0, len(sf)):
				if sf[i, start_time] == curr_time:
					curr_event.append(sf[i, idx ])
				else:
					timed_events.append(curr_event)

					soc_abbr_list = soc_abbr_list | set(curr_event)

					curr_time = sf[i, start_time]
					curr_event = []
					curr_event.append(sf[i, idx])

			if len(curr_event) >0 :
				timed_events.append(curr_event)		

			for i in range(0,len(timed_events)-lookahead+1):

				pt = []

				for j in range(0,lookahead) :
					pt.append(timed_events[i+j])
    
				data.append(pt)

		# self.data = data

		#generate one hot encodings of soc-abbreviations -> to convert them to vectors
		soc_abbr_list = list(soc_abbr_list)
		
		for i in range(0, len(soc_abbr_list)):
			self.soc_dict[soc_abbr_list[i]] = i
			self.inv_soc_dict[i] = soc_abbr_list[i]

		self.num_classes = len(self.soc_dict.keys())

		# print(soc_abbr_list)

		self.x_train, self.y_train, self.x_test, self.y_test = self.generate_data(data, dense, dimension)



