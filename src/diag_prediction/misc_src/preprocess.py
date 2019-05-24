import pandas as pd  
import pickle as pkl
import numpy as np
import random

train_data = [] 
look_ahead = 5

df = pd.read_csv('data/proact/AdverseEvents.csv')
df.loc[df['Outcome'] != 'Death','Outcome'] = 0
df.loc[df['Outcome'] == 'Death','Outcome'] = 1
df.loc[df['Lowest_Level_Term'] == 'Death','Outcome'] = 1

td = df[['subject_id','SOC_Abbreviation','Outcome']]
#td = pd.read_pickle('td.pkl')

x_train = []
y_train = []

soc_idx = 1
out_idx = 2

for name, group in td.groupby('subject_id'):
     
     if len(group) < 1 : 
          continue

     sf = group.values

     if len(sf) < look_ahead :
          continue 
          # pt = []
          
          # for j in range(0,min(look_ahead,len(sf))):
          #      pt.append(sf[j,soc_idx])
          #      pt.append(sf[j,out_idx])

          # train_data.append(pt)

     for i in range(0,len(sf)-look_ahead+1):

          outcome = 0
          pt = []

          for j in range(0,min(look_ahead-1,len(sf))) :
               pt.append(sf[i+j,soc_idx])

          pt.append(sf[i+look_ahead-1,soc_idx])    
          train_data.append(pt)

print('done')

#generate one hot encodings of soc-abbreviations -> to convert them to vectors
attr_list = list(td.SOC_Abbreviation.unique())
attr_dict = {}
for i in range(0, len(attr_list)):
     v = [0]*len(attr_list)
     v[i] = 1
     attr_dict[attr_list[i]] = v


#preparing the input
#70:30 split

random.shuffle(train_data)

train_size = int(0.70*len(train_data))

train_data = np.array(train_data)
train = train_data[:train_size]
test = train_data[train_size+1:]

train_l = train[:,-1]  
train_d = train[:,:-1]

time_steps = look_ahead - 1
for tra in train_d:
     tr_exp = []
     for i in range(0,len(tra)):
          tr_exp.append(attr_dict[tra[i]])
     x_train.append(tr_exp)

x_train = np.array(x_train)

#train labels
for tlabel in train_l:
     y_train.append(attr_dict[tlabel])

y_train = np.array(y_train)

#test labels
#test data

test_l = test[:,-1]  
test_d = test[:,:-1]

x_test = []
y_test = []

time_steps = look_ahead - 1
for tes in test_d:
     ts_exp = []
     for i in range(0,len(tes)):
          ts_exp.append(attr_dict[tes[i]])
     x_test.append(ts_exp)

x_test = np.array(x_test)

#test labels
for tlabel in test_l:
     y_test.append(attr_dict[tlabel])

y_test = np.array(y_test)

#have to pickle the data
f_xtr = open("x_train", "wb")
pkl.dump(x_train, f_xtr)
f_xtr.close()

f_ytr = open("y_train", "wb")
pkl.dump(y_train, f_ytr)
f_ytr.close()

f_xte = open("x_test", "wb")
pkl.dump(x_test, f_xte)
f_xte.close()

f_yte = open("y_test", "wb")
pkl.dump(y_test, f_yte)
f_yte.close()



