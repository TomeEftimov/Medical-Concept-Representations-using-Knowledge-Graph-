#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import pickle as pk

#Maps cui2vec embeddings from it's native (UMLS CUI based) format to
# SCUI -> Embeddings
class CUI2VecEmbeddingMapper:
    def __init__(self, cuivec_emb_file, cui_scui_file):
        df = pd.read_csv(cui_scui_file, sep="|", keep_default_na=False, dtype={'cuis': 'str',
                                                        'cuis_to_scuis' : 'str'})
        self.cui2scui = dict()
        for idx,row in df.iterrows():
            self.cui2scui[row['cuis']] = row['cuis_to_scuis']
        print("len of cui-scui dictionary", len(self.cui2scui))
        print(" dict(C0000052) ", self.cui2scui['C0000052'] )
        assert(self.cui2scui['C0000052'] == '58488005')

        miss = 0
        fout = open(cuivec_emb_file + ".scui_format.txt", "w")
        with open(cuivec_emb_file, "r") as f:
            header = f.readline()
            emb_dims = len(header.strip().split(",")) - 1
            fout.write('109053 ' + str(emb_dims) + "\n")
            for line in f:
                arr = line.strip().split(",")
                cui = arr[0]
                embeddings = arr[1:]
                if cui.startswith('"') and cui.endswith('"'):
                    cui = cui[1:-1]
                if (cui in self.cui2scui):
                    scui = self.cui2scui[cui]
                    if scui != 'nan' and scui != '':
                        str_embeddings = " ".join([str(x) for x in embeddings])
                        fout.write(str(scui) + " " + str_embeddings + "\n")
                    else:
                        miss += 1
        print("Number of cui misses/total", miss)
        return

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("Usage : <cui2vec_embeddings> <cui2scui_dict>")
        sys.exit(1)
    CUI2VecEmbeddingMapper(sys.argv[1], sys.argv[2])

