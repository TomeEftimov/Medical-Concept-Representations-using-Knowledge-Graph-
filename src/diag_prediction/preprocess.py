#preprocessing script for mimic3 dataset
import pandas as pd  
import pickle as pkl
import numpy as np
import random
import os.path
import sys

from mimic import mimic
from proact import proact

class Preprocess():

	x_train = []
	x_test = []
	y_train = []
	y_test  = []

	num_classes = 0


	def __init__(self, emb_file, lookahead=3, dataset='mimic', dense=False, dimension=32):
		
		if dataset == 'mimic':
			m = mimic(lookahead=lookahead, dense=dense, dimension=dimension, emb_file=emb_file)
			self.num_classes = m.num_classes
			self.x_train, self.y_train, self.x_test, self.y_test = m.x_train, m.y_train, m.x_test, m.y_test

		elif dataset == 'proact':
			p = proact(lookahead=lookahead, dense=dense, dimension=dimension)
			self.num_classes = p.num_classes
			self.x_train, self.y_train, self.x_test, self.y_test = p.x_train, p.y_train, p.x_test, p.y_test
