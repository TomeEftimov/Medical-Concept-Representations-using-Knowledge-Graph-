# file to load embeddings into a dictionary
import numpy as np

# embeddings = {}

# dim = 0
# with open('data/mimic3/emb.8') as femb:
# 	meta = femb.readline()
# 	meta.replace('\n','')
# 	x, dim = meta.split(' ')

# 	for line in femb:
# 		line.replace('\n','')
# 		node_emb = line.split(' ')
		
# 		node = int(node_emb[0])
# 		node_emb = np.array(node_emb[1:], dtype=float)

# 		embeddings[node] = node_emb

# # these two embeddings were not given. so, we initialize them with 0
# embeddings[181] = [0]*int(dim)
# embeddings[241] = [0]*int(dim)

class GraphEmbeddings():

	def getEmbeddings(self,dimension):
		embeddings = {}

		
		with open('data/mimic3/emb.'+str(dimension)) as femb:
			meta = femb.readline()
			meta.replace('\n','')
			x, dim = meta.split(' ')

			for line in femb:
				line.replace('\n','')
				node_emb = line.split(' ')
				
				node = int(node_emb[0])
				node_emb = np.array(node_emb[1:], dtype=float)

				embeddings[node] = node_emb

		# these two embeddings were not given. so, we initialize them with 0
		embeddings[181] = [0]*int(dim)
		embeddings[241] = [0]*int(dim)

		return embeddings

