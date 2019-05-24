# file to load embeddings into a dictionary
import numpy as np

class GraphEmbeddings():
	"""
	This class is used to load the embeddings which were obtained using node2vec or anything else
	It assumes each file has the following description
	number of nodes, dimension
	followed by each line containing vertex and embedding (only separated by spaces)
	"""
	def getEmbeddings(self, emb_file):
		embeddings = {}
		dimension = None
		femb = open(emb_file)
		meta = femb.readline()
		meta.replace('\n','')
		x, dimension = meta.split(' ')
		for line in femb:
			line.replace('\n','')
			node_emb = line.strip().split(' ')
			#node = int(node_emb[0])
			node = node_emb[0]
			node_emb = np.array(node_emb[1:], dtype=float)
			embeddings[node] = node_emb
		femb.close()
		return embeddings

	def getEmbeddings2(self, emb_file):
		embeddings = {}
		dimension = None
		femb = open(emb_file)
		meta = femb.readline()
		meta.replace('\n','')
		x, dimension = meta.split(' ')
		for line in femb:
			line.replace('\n','')
			node_emb = line.strip().split(' ')
			node = int(node_emb[0])
			node_emb = np.array(node_emb[1:], dtype=float)
			embeddings[node] = node_emb
		femb.close()
		# these embeddings were not given. so, we initialize them with 0
		for vertex in range(0,284):
			if vertex not in set(list(embeddings.keys())):
				embeddings[vertex] = [0]*int(dimension)
		return embeddings
