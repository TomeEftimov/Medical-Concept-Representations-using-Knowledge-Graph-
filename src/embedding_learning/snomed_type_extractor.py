#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import pickle as pk

# Processes type information for UMLS concept ids and generates type information
# for SNOMED_CUI 
class umls_snomed_type_extractor:
    """ Reads cui_scui.tsv and cui_tui.tsv to create scui_tui.tsv"""
    def __init__(self, snomed_dir):
        self.cui_scui = dict()
        self.cui_tui = dict()
        self.init_dict(self.cui_scui, snomed_dir + "/cui_scui.tsv")
        self.init_dict(self.cui_tui, snomed_dir + "/cui_tui.tsv")
        self.scui_tui = dict()
        self.merge_dict()
        self.dump_dict(snomed_dir + "/scui_tui.tsv")

    def init_dict(self, mydict, filename):
        with open(filename) as f:
            for line in f:
                arr=line.strip().split("\t")
                if(len(arr) == 2):
                    mydict[arr[0]] = arr[1]
                else:
                    print("Ignored line", line)

    def merge_dict(self):
        for cui, scui in self.cui_scui.iteritems():
            if(cui in self.cui_tui): 
                tui = self.cui_tui[cui]
                self.scui_tui[scui] = tui

    def dump_dict(self, outfile):
        with open(outfile, "w") as f:
            for k,v in self.scui_tui.iteritems():
                f.write(str(k) + "\t" + str(v) + "\n")


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage : <snomed_type_dict_dir>")
        sys.exit(1)
    umls_snomed_type_extractor(sys.argv[1])

