import sys
import os
import random
from collections import Counter

class MetaPathGenerator:
    """Snomed Adaptation for metapath2vec path generator
    1) We capture disease-drug interations through all possible paths 
    (direct and indirect). The 1-hop indirect paths are found through common 
    procedures and anatomies
    2) Generate random instances from sampling disease-drug-disease pathways

    Note: Metapath2vec code only supports four node types 'v', 'a', 'i' , 'f'.
    hence, we map snomed nodes as following:
         v -> disorder
         a -> anatomy
         i -> chemicals and drugs
         f -> procedure
    
    WORK EXTENSION: How do we extend the pattern generation code when we have several node
    types"""

    # load snomed graph by node type
    def __init__(self, snomed_dir):
        self.disorder_drugslist = dict()
        self.disorder_anatlist = dict()
        self.disorder_proclist = dict()
        self.drugs_disorderlist = dict()
        self.drugs_anatlist = dict()
        self.drugs_proclist = dict()
        self.anat_disorderlist = dict()
        self.anat_drugslist = dict()
        self.anat_proclist = dict()
        self.proc_disorderlist = dict()
        self.proc_drugslist = dict()
        self.proc_anatlist = dict()
        
        print("Getting concept types from", snomed_dir)
        self.disorders, self.drugs, self.anatomy, self.procedures = self.load_node_list(snomed_dir)
        print("Number of disorders, drugs, antomy, procedures",
              len(self.disorders), len(self.drugs), len(self.anatomy),
              len(self.procedures))
        self.init_graph(snomed_dir)
        print("Done initing graph and node types")
        print("Adj list size :", len(self.disorder_drugslist),
              len(self.drugs_disorderlist))

    def load_node_list(self, snomed_dir):
        snomed_graph_dir = snomed_dir + "/Unique_SNOMEDCT_concepts_per_semantic_group/"
        filename = snomed_graph_dir + "/mrconso_snomed_diso_unique.txt"
        disoders = self.parse_node_list_file(filename)
        filename = snomed_graph_dir + "/mrconso_snomed_chem_unique.txt"
        drugs = self.parse_node_list_file(filename)
        filename = snomed_graph_dir + "/mrconso_snomed_anat_unique.txt"
        anatomy = self.parse_node_list_file(filename)
        filename = snomed_graph_dir + "/mrconso_snomed_proc_unique.txt"
        procedures = self.parse_node_list_file(filename)
        return disoders, drugs, anatomy, procedures

    def parse_node_list_file(self, filename):
        concept_list = []
        with open(filename, "r") as f:
            header = f.readline()
            for line in f:
                arr = line.strip().split('|')
                # Use snomed concept ids
                concept = arr[1] #arr[4].replace(" ", "")
                concept_list.append(concept)
        return concept_list

    def init_graph(self, snomed_dir):
        filename = snomed_dir + "/SNOMEDCT_relations.txt"
        print("Reading graph  from...", filename)
        with open(filename, "r") as f:
            header = f.readline()
            for line in f:
                arr = line.strip().split('|')
                src = arr[1] #arr[2].replace(" ", "")
                dest = arr[3] #arr[4].replace(" ", "")
                relation = arr[5].replace(" ", "")
                if src in self.disorders:
                    self.add_disorder(src, dest)
                elif src in self.drugs:
                    self.add_drugs(src, dest)
                #elif src in self.anatomy:
                #    self.add_anatomy(src, dest)
                #elif src in self.procedures:
                #    self.add_procedure(src, dest)
                else:
                    continue
        return
    
    #add the edge iff of type disorder -> {drugs | procedure | anatomy}
    def add_disorder(self, src, dest):
        if dest in self.drugs: 
            if(src not in self.disorder_drugslist):
                self.disorder_drugslist[src] = []
            self.disorder_drugslist[src].append(dest) 
        #elif dest in self.anatomy:
        #    if(src not in self.disorder_anatlist):
        #        self.disorder_anatlist[src] = []
        #    self.disorder_anatlist[src].append(dest)
        #elif dest in self.procedures:
        #    if(src not in self.disorder_proclist):
        #        self.disorder_proclist[src] = []
        #    self.disorder_proclist[src].append(dest)
        else: #if dest in self.disorders:
            return
        #else:
        #    print("In add_disorder() Found a node of unknown type", dest)
        return

    #add the edge iff of type drugs->disorder/proc/anatomy
    def add_drugs(self, src, dest):
        if dest in self.disorders:
            #print("Adding drugs->disorder", src, dest)
            if(src not in self.drugs_disorderlist):
                self.drugs_disorderlist[src]  = []
            self.drugs_disorderlist[src].append(dest)
        #elif(dest in self.procedures):
        #    if(src not in self.drugs_proclist):
        #        self.drugs_proclist[src] = []
        #    self.drugs_proclist[src].append(dest)
        #elif(dest in self.anatomy):
        #    if(src not in self.drugs_anatlist):
        #        self.drugs_anatlist[src] = []
        #    self.drugs_anatlist[src].append(dest)
        else: #if(dest in self.drugs):
            return

    #add the edge iff of type procedure->disorder/prioc/anatomy
    def add_procedure(self, src, dest):
        if dest in self.disorders:
            if(src not in self.proc_disorderlist):
                self.proc_disorderlist[src] = []
            self.proc_disorderlist[src].append(dest)
        elif dest in self.drugs:
            if(src not in self.proc_drugslist):
                self.proc_drugslist[src] = []
            self.proc_drugslist[src].append(dest)
        elif dest in self.anatomy:
            if(src not in self.proc_anatlist):
                self.proc_anatlist[src] = []
            self.proc_anatlist[src].append(dest)
        else: # dest in self.procedures or another node type
            return
    
    #add the edge iff of type anatomy->disorder
    def add_anatomy(self, src, dest):
        if dest in self.disorders:
            if(src not in self.anat_disorderlist):
                self.anat_disorderlist[src] = []
            self.anat_disorderlist[src].append(dest)
        elif dest in self.drugs:
            if(src not in self.anat_drugslist):
                self.anat_drugslist[src] = []
            self.anat_drugslist[src].append(dest)
        elif dest in self.procedures:
            if(src not in self.anat_proclist):
                self.anat_proclist[src] = []
            self.anat_proclist[src].append(dest)
        else: # dest in self.anatomy:
            return

    #def update_disorder_and_drugs_adj(self):
    #    for procedure in procedure_disorderlist.keys():
    #        if procedure in procedure_drugslist.keys():


    # generate initial path patterns around disorder, since thats our main input
    # to patient model:
        # P1 : Disorder, drug, disoder
        # P2 : Disorder, procedure, disorder
        # P3 : Disorder, anatomy, disorder
        # Mapping snomed nodes as following:
        #   v -> disorder
        #   a -> anatomy
        #   i -> chemicals and drugs
        #   f -> procedure
        # To generate path P1: Load all disorders and their nbrs, load all drugs
        # and their disorder nbrs. sample from the neighbourhood like ACA
        # path generation

    def generate_paths(self, outfilename, numwalks, walklength):
        self.generate_random_concept1_to_concept2_paths(outfilename,
                                                        self.disorder_drugslist,
                                                        self.drugs_disorderlist,
                                                        'v', 'i', numwalks,
                                                        walklength)


    def generate_random_concept1_to_concept2_paths(self, outfilename,
                                            c1_to_c2,
                                            c2_to_c1,
                                            c1_prefix_letter, c2_prefix_letter,
                                            numwalks, walklength):
        outfile = open(outfilename, "w")
        #print("Walking disoredrs", self.disorders)
        #for disorder in self.disorders:
        all_valid_c1 = list(c1_to_c2.keys())
        for c1 in all_valid_c1:
            c1_0 = c1
            for j in range(0, numwalks):
                outline = c1_prefix_letter + c1_0
                for i in range(0, walklength):
                    if c1 in c1_to_c2:
                        all_valid_c2 = c1_to_c2[c1]
                        c2 = random.choice(all_valid_c2)
                        if c2 in c2_to_c1:
                            valid_c1s = c2_to_c1[c2]
                            outline = outline + " " + c2_prefix_letter + c2
                            c1 = random.choice(valid_c1s)
                            outline = outline + " " + c1_prefix_letter + c1
                        else:
                            continue
                    else:
                        continue
                if(len(outline.split(" ")) >= walklength):
                    outfile.write(outline+ "\n")
                else:
                    print("Could not complete walk, ignoring ", outline)
            outfile.flush()
        outfile.close()
        return

    #Generate {author->list(conference)} and {conference->list(authors)} from
    #author->paper and paper->conference edges
    #def generate_random_aca(self, outfilename, numwalks, walklength):
    #    for conf in self.conf_paper:
    #        self.conf_authorlist[conf] = []
	#		 for paper in self.conf_paper[conf]:
	#			if paper not in self.paper_author: continue
	#			for author in self.paper_author[paper]:
	#				self.conf_authorlist[conf].append(author)
	#				if author not in self.author_conflist:
	#					self.author_conflist[author] = []
	#				self.author_conflist[author].append(conf)
	#	#print "author-conf list done"

	#	outfile = open(outfilename, 'w')
        # For every conference : generate numwalks of pattern "ACA.."
	#	for conf in self.conf_authorlist:
	#		conf0 = conf
	#		for j in xrange(0, numwalks ): #wnum walks
	#			outline = self.id_conf[conf0]
                #start walk with aconference and generate
                # conf -> author -> conf walks
#				for i in xrange(0, walklength):
#					authors = self.conf_authorlist[conf]
#					numa = len(authors)
#					authorid = random.randrange(numa)
#					author = authors[authorid]
#					outline += " " + self.id_author[author]
#					confs = self.author_conflist[author]
#					numc = len(confs)
#					confid = random.randrange(numc)
#					conf = confs[confid]
#					outline += " " + self.id_conf[conf]
#				outfile.write(outline + "\n")
#		outfile.close()


#python py4genMetaPaths.py 1000 100 net_aminer output.aminer.w1000.l100.txt
#python py4genMetaPaths.py 1000 100 net_dbis   output.dbis.w1000.l100.txt

dirpath = "snomed" 
# OR 
dirpath = "net_dbis"

#takes as input path to snomed graph dir (containing umls graph) and the
#numwalks, walklength. Gernerates pattern walk for "disorder->drug->disorder"
numwalks = int(sys.argv[1])
walklength = int(sys.argv[2])

dirpath = sys.argv[3]
outfilename = sys.argv[4]

def main():
	mpg = MetaPathGenerator(dirpath)
	mpg.generate_paths(outfilename, numwalks, walklength)

if __name__ == "__main__":
	main()






























