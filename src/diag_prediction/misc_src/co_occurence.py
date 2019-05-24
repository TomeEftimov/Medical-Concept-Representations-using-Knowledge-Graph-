
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys


# In[2]:


df = pd.read_csv('data/mimic3/DIAGNOSES_ICD.csv')


# In[3]:

# In[4]:


import pickle as pkl
fh = open('inv_idx','rb')
inv_idx = pkl.load(fh)
fh.close()

inv_idx['nan'] = 284
adj = [[0]* 284]*284


# In[5]:


count = 0
td = df[['HADM_ID','ICD9_CODE']]
for name, group in td.groupby('HADM_ID'):
    count += 1
    codes = group['ICD9_CODE'].values
    codes = [inv_idx[str(code)] for code in codes]
    for i in range(0,len(codes)):
        for j in range(i+1,len(codes)):
            x,y = codes[i], codes[j]
            adj[x][y] += 1
            adj[y][x] += 1

#print(count)


# In[6]:


adj_c = []
for i in range(0,len(adj)):
    for j in range(0,len(adj[i])):
        if adj[i][j] == 0:
            continue
        else:
            adj_c.append(adj[i][j])


# In[7]:


#len(adj_c)


# In[11]:


adj_c.sort()
adj_c = np.array(adj_c)
Delta = np.mean(adj_c)


# In[13]:


l = [val for val in adj_c if val > Delta]
#print(len(l))

# In[15]:


coG = [[0]*284]*284
for i in range(0,284):
    for j in range(0,284):
        if adj[i][j] > Delta:
            coG[i][j] = 1
        else:
            coG[i][j] = 0
            


# # In[ ]:


num_classes = 284
for i in range(0,(num_classes)):
	for j in range(0,(num_classes)):
		if coG[i][j] == 1 and not i==j:
			print(str(i)+"\t"+str(j))

