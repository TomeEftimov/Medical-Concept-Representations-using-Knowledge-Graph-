import pandas as pd  
import pickle as pkl
import numpy as np
import random
import os.path
import sys

from embeddings import GraphEmbeddings

class mimic():

    """
    We haven't pickled the files for train, test. So, with default settings, a call is made to 
    mimic.py to prepare necessary data. The relevant parameters here are (dense, dimension, lookahead).

    When *dense=False*, the objects returned are train and test data. 
    Train data contains examples for training and their labels.
    x_train contains examples used for training and is a N x (lookahead-1) x 284 numpy array. 
    y_train contains labels for corresponding examples used. It is a N x 284 numpy array.

    Test data contains examples for testing and their labels.
    x_test contains examples used for testing and is a M x (lookahead-1) x 284 numpy array. 
    y_test contains labels for corresponding examples used. It is a M x _ numpy array. 
    During training, our labels are encoded in a multi-hot way. Because, we used our own accuracy function, we defined test labels for convenience as follows. Each test label is basically a list of diagnosis entries for eg., [100, 200, 223].

    When *dense=True*, we also give the dimension for the embeddings the objects returned are train and test data.
    Train data contains examples for training and their labels.
    x_train contains examples used for training and is a N x lookahead-1 x dimension numpy array. 
    y_train contains labels for corresponding examples used. It is a N x 284 numpy array.

    Test data contains examples for testing and their labels.
    x_test contains examples used for testing and is a M x lookahead-1 x dimension numpy array. 
    y_test contains labels for corresponding examples used. It is a M x _ numpy array. 
    During training, our labels are encoded in a dense way. Because, we used our own accuracy function, we defined test labels for convenience as follows. Each test label is basically a list of diagnosis entries for eg., [100, 200, 223].

    Some relevant pickled files in the repo : 
    adm_dt : for every hospital admission, we extract date time object and pickle it
    inv_idx : every icd9 code is mapped to a unique id between 0-283 based on classification.
    
    """
    adm_dt = {}
    co_G = []
    vertexEmb = None

    x_train = []
    x_test = []
    y_train = []
    y_test  = []

    num_classes = 0


    def get_code(self, code):
        if len(code) <= 3:
            code = str(code)
        if code[0] =='V':
            code = code[0:3]+'.'+code[3:]
        elif code[0] =='E':
            code = code[0:4]+'.'+code[4:]
        else:
            code = code[0:3]+'.'+code[3:]

        return str(code)

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
                
                pt_exp = [0]*284
                for j in range(0,len(point[i])):
                    pt_exp[point[i][j]]= 1

                seq.append(pt_exp)

            embed_data.append(seq)
        
        return np.array(embed_data)

    def generate_dense_enc(self, data, dimension ):
        
        embed_data = []
        embeddings = self.embeddings
        embed_set = set(list(embeddings.keys()))
        #print("Embeddings available for", embed_set)

        # fh = open('code_dict','rb')
        # code_dict = pkl.load(fh)
        # fh.close()

        # inv_code_dict = {}
        # for key, value in code_dict.items():
        #   inv_code_dict[value] = key 

        # For each sample (patient)
        missing_diagnosis_codes = 0
        total_diagnosis_codes = 0
        for pt in data:
            pt_exp = []
            
            # For each visit
            for i in range(0,len(pt)):
                v = np.zeros(dimension)
                total_diagnosis_codes += len(pt[i])
                #print("Looking for snomed diagnosis codes", pt[i])

                # For each diagnosis code
                for j in range(0,len(pt[i])):
                    # code_vertex = code_dict[pt[i][j]]
                    # code = code_vertex
                    # code = self.get_code(str(pt[i][j]))

                    code = pt[i][j]
                    if code in embed_set:
                        v = np.add(v, np.array( embeddings[code] ))
                    else:
                        #print("Missing embeddings for ", code)
                        missing_diagnosis_codes +=1

                # v = v/len(pt[i])
                pt_exp.append(v)

            embed_data.append(pt_exp)
        print("Embedding coverage for snomed (missing, total)", missing_diagnosis_codes, total_diagnosis_codes) 
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
            self.embeddings = self.vertexEmb.getEmbeddings(self.emb_file)
            # with open('snomed_embeddings.pkl','rb') as f:
                # self.embeddings = pkl.load(f)

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

    def get_neighbors(self, neighbors, icd9_codes):
        for code in icd9_codes:
            icd9_codes = icd9_codes + list(neighbors[code])

        return list(set(icd9_codes))

    def __init__(self, lookahead, dense, dimension, emb_file):
        self.emb_file = emb_file
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
        inv_idx_pred_labels = pkl.load(fh)
        fh.close()

        fh = open('icd_snomed.dict.pkl','rb')
        inv_idx_diagnosis_codes = pkl.load(fh)
        fh.close()

        fdt = open('adm_dt','rb')
        self.adm_dt = pkl.load(fdt)
        fdt.close()

        fnbr = open('misc_src/neighbors','rb')
        neighbors = pkl.load(fnbr)
        fnbr.close()

        inv_idx_pred_labels['nan'] = 283

        uless_count = 0
        uful_count = 0
        count_sub = 0
        total_diag_count = 0
        unique_icd9_codes = set([])
        count_icd9_codes = {}

        data = []
        total_missing_diagnosis_codes = 0
        total_diagnosis_codes = 0
        for name, group in td.groupby('SUBJECT_ID'):    

            l = group.groupby(['HADM_ID'])
                        # l is list of visits
            if len(l) >= lookahead:
                
                sub = group['SUBJECT_ID'].values[0]
                hadm_code_dict = {}
                hadm_list = []

                for c in l:
                    hadm_list.append(c[0])
                    # Note : Depending on inv_idx, these should be mapped to SCUI or the 284 icd codes
                    icd9_codes = [str(x).replace('"', '') for x in c[1]['ICD9_CODE'].values]

                    #icd9_codes = [self.get_code(str(x)) for x in c[1]['ICD9_CODE'].values]

                    # icd9_codes = [str(x) for x in c[1]['ICD9_CODE'].values]
                    
                    hadm_code_dict[c[0]] = icd9_codes

                hadm_list.sort(key=self.cmp_to_key(self.compare_datetime))
                
                for i in range(0,len(hadm_list)-lookahead+1):
                    
                    seq = []
                    #Input sample (with length(input) = #lookahead)  
                    for j in range(0, lookahead-1):
                        # seq.append(self.get_neighbors(neighbors,hadm_code_dict[hadm_list[i+j]]))
                        visit_codes = hadm_code_dict[hadm_list[i+j]]
                        if dense == True:
                            # Use dense embeddings ffrom Snomed
                            missing_codes = 0
                            scui_visit_codes = []
                            for x in visit_codes:
                                if x in inv_idx_diagnosis_codes:
                                    scui_visit_codes.append(inv_idx_diagnosis_codes[x])
                                else:
                                    missing_codes += 1
                                    #print("Could not dind icd_code in snomed dict", x) 
                                total_missing_diagnosis_codes += missing_codes
                                total_diagnosis_codes += len(visit_codes)
                            #scui_visit_codes = [inv_idx_diagnosis_codes[x] for x in visit_codes if x in inv_idx_diagnosis_codes]
                        else:
                            # use multi hot 284 icd codes
                            scui_visit_codes = [inv_idx_pred_labels[x] for x in visit_codes]
                        #seq.append(hadm_code_dict[hadm_list[i+j]])
                        seq.append(scui_visit_codes)
                                        
                    # Output (Objective) per sample per patient
                    visit_codes = hadm_code_dict[hadm_list[i+lookahead-1]]
                    icd_group_codes = [inv_idx_pred_labels[x] for x in visit_codes]
                    seq.append(icd_group_codes)
                    #seq.append(hadm_code_dict[hadm_list[i+lookahead-1]])
                        
                    # seq.append([inv_idx[str(x)] for x in hadm_code_dict[hadm_list[i+lookahead-1]]])

                    data.append(seq)

        if(dense):
            print("Coverage of icd to snomed diagnosis code (missing, total)",
                  total_missing_diagnosis_codes , total_diagnosis_codes)
        # self.attr_list = list(unique_icd9_codes)
        self.num_classes = 284
        random.shuffle(data)
        self.x_train, self.y_train, self.x_test, self.y_test = self.generate_data(data, dense, dimension)
        # print('-------------- Data is Ready!! --------------')
