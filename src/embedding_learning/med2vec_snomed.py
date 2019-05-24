#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import pickle as pk

#Reads med2vec embeddingsh in its native (ICD-9 based)format and generates
# SCUI -> EMbeddings
class Med2VecEmbeddingMapper:
    def __init__(self, med2vec_emb_file, med2vec_int2str, icd_scui_pkl_file):
        med2vec_dict = pk.load(open(med2vec_int2str, "rb"))
        embeddings = np.load(med2vec_emb_file)
        
        icd_scui_dict = pk.load(open(icd_scui_pkl_file, "rb"))
        
        fout = open(med2vec_emb_file + ".scui_format.txt", "w")
        fout.write(str(len(embeddings)) + " " + str(len(embeddings[0])) + "\n")
        miss = 0
        for i in range(0, len(embeddings)):
            med2vec_str = med2vec_dict[i]
            #Strip first 2 letters from med2vec icd9 code as it encodes the node
            #type information
            icd_9 = med2vec_str.replace('.', '')
            if icd_9.startswith("D_") or icd_9.startswith("P_") or icd_9.startswith("R_"):
                icd_9 = icd_9[2:]
            else:
                print("Found an ICD9 without prefix")
            if (icd_9 in icd_scui_dict):
                scui = icd_scui_dict[icd_9]
                str_embeddings = " ".join([str(x) for x in embeddings[i]])
                fout.write(str(scui) + " " + str_embeddings + "\n")
            else:
                #print("Could not find ICD-9 code in ICD_SNOMED map: ",icd_9)
                miss += 1
        print("Number of misses/total", miss, len(embeddings))
        return

if __name__ == "__main__":
    if(len(sys.argv) != 4):
        print("Usage : <med2vec_embeddings> <med2vec_dict> <icd_snomed.pkl>")
        sys.exit(1)
    Med2VecEmbeddingMapper(sys.argv[1], sys.argv[2], sys.argv[3])

