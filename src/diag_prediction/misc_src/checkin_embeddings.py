# file to load embeddings into a dictionary
import numpy as np

class GraphEmbeddings():

	def getEmbeddings(self,dimension):
		
		embeddings = {}
		with open('data/mimic3/emb.sparse3.'+str(dimension)) as femb:
			meta = femb.readline()
			meta.replace('\n','')
			x, dim = meta.split(' ')

			for line in femb:
				line.replace('\n','')
				node_emb = line.split(' ')
				
				node = int(node_emb[0])
				node_emb = np.array(node_emb[1:], dtype=float)

				embeddings[node] = node_emb

		# these embeddings were not given. so, we initialize them with 0
		for vertex in range(0,284):
			if vertex not in set(list(embeddings.keys())):
				embeddings[vertex] = [0]*int(dimension)

		return embeddings

