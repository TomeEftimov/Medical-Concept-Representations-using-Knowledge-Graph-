#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import pickle as pk
# load umls nomed graph as list of SCUI edges between(anat, disorder, drugs,procedures) 
# load vertex_dict for SCUI
# create a node2vec format graph using : SCUI vertex dictionary, SCUI edges 

class umls_snomed_extractor:
    """Reads snomed_relations file to get "scui <-> scui" relations
    1) creates the graph in following format
    scui_normalized_id <-> scui_normalized_id (for node2vec 0 to N-1 id format)
    2) dump vertex_dict for 
    scui -> normalized_id
    """
    def __init__(self, filename, outdir):
        self.scui_id = dict()
        self.edges = list()
        self.num_scui_nodes = 0
        self.init_graph(filename)
        self.dump_normalized_graph(outdir + "/umls_snomed_scui.int.tsv")
        self.dump_vertex_dict(outdir + "/vertex_dict.tsv")

    def init_graph(self, filename, sep= "|"):
        with open(filename) as f:
            header = f.readline()
            for line in f:
                arr = line.strip().split(sep)
                if(len(arr) != 6 or arr[1] == "" or arr[3] == ""):
                    print("Unexpected format, ignored :", line)
                    continue
                src_scui = arr[1]
                dest_scui = arr[3]
                relation = arr[5]
                self.edges.append((arr[1], arr[3]))
                self.add_vertex(arr[1])
                self.add_vertex(arr[3])
    
    def add_vertex(self, scui):
        if(scui not in self.scui_id):
            self.scui_id[scui] = self.num_scui_nodes
            self.num_scui_nodes = self.num_scui_nodes + 1
        return

    def dump_vertex_dict(self, outfile):
        with open(outfile, "w") as f:
            for k,v in self.scui_id.iteritems():
                f.write(str(k)+ "\t" + str(v) + "\n")

    def dump_normalized_graph(self, outfile):
        fout = open(outfile, "w")
        for edge in self.edges:
            src_id = self.scui_id[edge[0]]
            dest_id = self.scui_id[edge[1]]
            fout.write(str(src_id) + "\t" + str(dest_id) + "\n")
        fout.close()

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


class InvEmbeddingMapper:
    # reads inverted vertex dictionary {int_id : graph_node}
    # converts {int_id : embedding} to {graph_node : embedding}
    def __init__(self, vertex_dict_file, int_id_emb_file):
        mydict = dict()
        with open(vertex_dict_file) as f:
            for line in f:
                arr = line.strip().split("\t")
                mydict[arr[1]] = arr[0]
        self.map_int_emb_to_orig_graph_vertices(int_id_emb_file, mydict)


    # takes the inout file embeddings in {int_id : embeddings}
    # generates {graph_node : embeddings} using self.vertex_dict
    # Note : can be use as a genric mapping function
    def map_int_emb_to_orig_graph_vertices(self, filename, vertex_dict):
        print("Size of vertex dictionary", len(vertex_dict))
        fout = open(filename + ".graph_format.txt", "w")
        with open(filename, "r") as f:
            header = f.readline()
            fout.write(header)
            for line in f:
                arr = line.strip().split(' ')
                graph_node = vertex_dict[arr[0]]
                str_emb = " ".join(x for x in arr[1:])
                fout.write(str(graph_node) + " " + str_emb + '\n')

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        #print("Usage : <snomed_relations file> <outdir>")
        print("Usage : <vertex_dict> <node2vec embedidngs file>")
        sys.exit(1)
    #snomed_obj = umls_snomed_extractor(sys.argv[1], sys.argv[2])
    InvEmbeddingMapper(sys.argv[1], sys.argv[2])

