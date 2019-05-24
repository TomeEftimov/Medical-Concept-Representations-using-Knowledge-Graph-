#!/usr/bin/python

import sys, io
#import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import sys, io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#from sklearn.metrics import precision_recall_fscore_support as metricScores

# reads embedding file in following format
## nodeid;;;;[float_array separated by ', ']
## nodeid;;;;[float_array separated by ', ']
def get_features_wordnet(featureFile):
    f = open(featureFile, "r")
    idToFeatureMap = {}
    for line in f:
        arr = line.strip().split(";;;;")
        if(len(arr) != 2):
            print("Could not parse line corrrectly, expecting id;;;;features perline")
            sys.exit()
        else:
            nodeid = int(arr[0])
            tempfeatures = arr[1].strip()
            # Remove the trailing '[' and ']' and split
            features = tempfeatures[1:-1].split(", ")
            # remove the trailing ' and ' from each floating and get the array
            emb = np.array([float(x[1:-1]) for x in features], dtype=np.float)
            idToFeatureMap[nodeid] = emb
    f.close()
    return idToFeatureMap

# reads embedding file where every line is in format
# emb [ array of floats separated by space]
def get_features_lstm(featureFile):
    f = open(featureFile, "r")
    #ln = f.readline()
    i = 0
    idToFeatureMap = {}
    for line in f:
        arr = line.strip().split(" ")
        emb = np.array(arr, dtype=np.float)
        idToFeatureMap[i] = emb
        i = i+1
    f.close()
    return idToFeatureMap


# reads embedding file in following format
## NUMNODES EMBSIZE
## nodeid1 float_array separated by space
## nodeid2 float_array separated by space
## ..
def get_features_node2vec(featureFile):
    f = open(featureFile, "r")
    ln = f.readline()
    arr = ln.strip().split(' ')
    if(len(arr) != 2):
        print("Could not read numlines, embedding size")
    numNodes = int(arr[0])
    embSize = int(arr[1])
    idToFeatureMap = {}
    for line in f:
        idWithEmb = line.strip().split(" ")
        if(len(idWithEmb) != embSize+1):
            print("Unable to parse feature file correctly, expecting id emb")
            sys.exit
        id = idWithEmb[0].strip()
        emb = np.array(idWithEmb[1:], dtype=np.float)
        idToFeatureMap[id] = emb
    f.close()
    return idToFeatureMap


## reads a file in follwing format per line
## nodeid;;;;nodeLabel1;;nodeLabel2;;nodeLabel3..
## returns map[nodeid -> list[labels] )
def get_node_labels_wordnet(nodeToLabelFile):
    f = open(nodeToLabelFile, "r")
    idLabelMap = {}
    for line in f:
        idWithLabel = line.strip().split(";;;;")
        if(len(idWithLabel) != 2):
            print("Unable to parse label file")
            exit
        id = int(idWithLabel[0])
        labels = idWithLabel[1].strip().split(";;")
        idLabelMap[id] = labels
        
    f.close()
    print("nodeid to label dictionary size", len(idLabelMap))
    return idLabelMap

## reads a file in follwing format per line
## node<splitter>node_class_label
## returns map[node -> list[labels] )
##NOte a node may be repetaed multiple times if it has multiple labels
def get_node_labels(labelFile, splitter="\t"):
    f = open(labelFile, "r")
    idLabelMap = {}
    for line in f:
        idWithLabel = line.strip().split(splitter)
        if(len(idWithLabel) != 2):
            print("Unable to parse label file line : ", line)
        else:
            node_label = idWithLabel[0].strip()
            node_class_label = idWithLabel[1].strip()
            idLabelMap[node_label] = node_class_label 
		    #id = int(idWithLabel[0].strip())
            #label= int(dWithLabel[1].strip())
		    #currLabels = []
		    #if(id in idLabelMap):
		    #	currLabels = idLabelMap[id]
		    #currLabels.append(node_class_label)
		    #idLabelMap[id] = currLabels
    f.close()
	#fout = open(labelFile + ".idLabelMap", "w+")
	#for key,value in idLabelMap.items():
	#	fout.write(str(key) + "\t" + str(value) + "\n")
	#fout.close()
    return idLabelMap

## takes two maps (nodeid -> features), (nodeid -> labels)
## merges maps assuming all nodes with labels have features available
## nodeid can be any integres (NOT restricetd to 0 to N-1)
## returns map (nodeid -> (feature, labels))
def merge_maps(idfeatureMap, idLabelMap):
    #print_sample_dict(idfeatureMap)
    #print_sample_dict(idLabelMap)
    mergedMap = {}
    for id, labels in idLabelMap.items():
        if(id in idfeatureMap):
            feature = idfeatureMap[id]
            mergedMap[id] = (feature, labels)
			#mergedMap[id] = (feature, labels[0])
		#else:
		#	print("Found a node without corresponding embedding", id)
    print("len of merged map", len(mergedMap))
    return mergedMap

"""Read a dict file in format :
    node_label<tab>node_id
    Return: {Nodeid:node_label}"""
def get_vertex_dict(filename):
    mydict = {}
    with open(filename, "r") as f:
        for line in f:
            arr = line.strip().split('\t')
            if (len(arr) == 2):
                mydict[arr[1]] = arr[0]
    print("Length of vertex dictionary : ", len(mydict))
    return mydict

def print_sample_dict(mydict):
    print_count = 5
    i = 0
    for k,v in mydict.items():
        if(i < 5):
            print(k,v)
        else:
            return
        i=i+1
		
def merge_maps_snomed(id_embed_map, node_label_to_node_class_map,
                      vertex_id_to_label_dict):
    label_to_id_map = {}
    for id, label in vertex_id_to_label_dict.items():
        label_to_id_map[label] = id

    id_to_node_class_map = {}
    # Note certain nodes may have their node types available but may not be part
    # of areduced version of the graph
    for node_label, node_class in node_label_to_node_class_map.items():
        if node_label in label_to_id_map:
            node_id = label_to_id_map[node_label]
            id_to_node_class_map[node_id] = node_class
    return merge_maps(id_embed_map, id_to_node_class_map)

def get_data(embed_file, node_class_label_file, vertex_dict_file):
    ## read map(nodeid->list(class_labels))
    labelMap = get_node_labels(node_class_label_file)
    ## read map(nodeid -> embedding_features)
    featureMap = get_features_node2vec(embed_file)
    print("Length of id->?emebdding map", len(featureMap)) 
    labelDict = {}
    all_labels = list(labelMap.values())
    i=0
    for v in  all_labels:
        if v not in labelDict:
            labelDict[v] = i 
            i = i + 1
    print ("Labeldict length", len(labelDict))

    ## merge maps to create {nodeid-> (features, labels)}
    ## note nodeid's are NOT guaranteed to be 0 to N-1
    ## merged map only contains nodeids that have labels and features
    ## Optional for node2vec : Get node mapping , ids from 0 to numLabels-1
    if vertex_dict_file != "NONE":
        vertex_dict = get_vertex_dict(vertex_dict_file)
        mergedMap = merge_maps_snomed(featureMap, labelMap, vertex_dict)
    else:
        mergedMap = merge_maps(featureMap, labelMap)
    print(" Length of merged map with embdinngs and node types: ",
          len(mergedMap)) 
    #mergedMap = merge_maps(featureMap, labelMap)
    featuresWithlabels = mergedMap.values()
    features = [x[0] for x in featuresWithlabels]
    labels = [convertToSparseMatrix_singleLabel(x[1], labelDict) for x in featuresWithlabels]
    #print(features[0], labels[0])
    #print(features[-1], labels[-1])
    return (np.array(features, dtype=np.float), np.array(labels))

##Reads a file containign all possible class labels and return a dictionary
## mapping each label to an id 0 to numLabels-1
def getLabelToIdDict(labelListFile):
    #First map each node label in id 0 to NLabels -1
    flabel = open(labelListFile, "r")
    i=0
    labelDict = {}
    for line in flabel:
        if(not line.startswith("#")):
            arr = line.strip().split(",")
            if(len(arr) >= 1):
                labelDict[arr[0]] = i
            else:
                print("could not recognize labelFile format", labelListFile)
                exit
            i= i+1
    flabel.close()
    print("Number of unique labels in label to id dictionary:", len(labelDict))
    fout = open(labelListFile + ".labelToLabelIdMap", "w+")
    for key,value in labelDict.items():
        fout.write(str(key) + "\t" + str(value) + "\n")
    fout.close()
    return labelDict
    
def convertToSparseMatrix(nodeLabels, labelDict):
    numNodeLabels = len(nodeLabels)
    numLabels = len(labelDict)
    sparseNodelabels = np.zeros(numLabels, dtype=np.int)
    for i in range(0, numNodeLabels):
        nodelabel = nodeLabels[i]
        labelid = labelDict[nodelabel]
        sparseNodelabels[labelid] = 1
    #print("Sparsifyoing node labels", nodeLabels, sparseNodelabels)
    return sparseNodelabels

def convertToSparseMatrix_singleLabel(nodeLabel, labelDict):
    numLabels = len(labelDict)
    sparseNodelabels = np.zeros(numLabels, dtype=np.int)
    labelid = labelDict[nodeLabel]
    sparseNodelabels[labelid] = 1
    #print("Sparsifyoing node labels", nodeLabels, sparseNodelabels)
    return sparseNodelabels

def run_classifier(embed_file, node_class_label_file, vertex_dict_file, test_percentage):
    print("Getting Data")
    (features, labels) = get_data(embed_file, node_class_label_file,
                                         vertex_dict_file)
    print(features.shape, labels.shape)
    numUniqueLabels = len(labels[0])
    print("Creating classifier")
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    X, xTest, Y, yTest = train_test_split(features, labels, test_size=test_percentage,random_state=0)
    print("trainign data shape", X.shape, Y.shape)
    print("Testing data shape", xTest.shape, yTest.shape)
    sys.stdout.flush()
    yScore = clf.fit(X, Y).decision_function(xTest)
    joblib.dump(clf, embed_file + '.clfmodel.pkl') 
    yPred = clf.predict(xTest)
    print("Actual vs Predicted class for each test sample")
    print(yTest[0:10], yPred[0:10])
    print("Fitting completed, scores = ")
    sys.stdout.flush()
    printScores(yTest, yScore, numUniqueLabels)

def printScores(yTest, yScore, numUniqueLabels):
    # Compute Precision-Recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(numUniqueLabels):
        precision[i], recall[i], _ = precision_recall_curve(yTest[:, i], yScore[:, i])
        average_precision[i] = average_precision_score(yTest[:, i], yScore[:, i])
    ## Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(yTest.ravel(),yScore.ravel())
    average_precision["micro"] = average_precision_score(yTest, yScore,average="micro")
    print("Average Precison (all classes) = ", average_precision)
    print("Precision(micro)", precision["micro"]) 
    print("Recall(micro)", recall["micro"]) 
    print("Average precision micro", average_precision["micro"]) 
    sys.stdout.flush()

def createFeatureFileWithId(tensorflowEmbFile, featureFileWithId, embSize, numNodes):
    fout = open(featureFileWithId, "w+")
    fout.write(str(numNodes) + " "  + str(embSize) + "\n")
    f = open(tensorflowEmbFile, "r")
    i = 0
    for line in f:
        fout.write(str(i) + " " + line)
        i = i+1
    f.close()
    fout.close()

if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print("Usage <tensorflowEmbFile> <nodeClassLabelFile> <testPercent> <vertex_dict(optional for node2vec)")
        exit()
    embed_file = sys.argv[1]
    node_class_labels_file = sys.argv[2]
    test_percentage = float(sys.argv[3])
    vertex_dict_file = "NONE" 
    if len(sys.argv) == 5:
        vertex_dict_file = sys.argv[4] 
    print("Running classifier with", embed_file, node_class_labels_file, vertex_dict_file, test_percentage)
    run_classifier(embed_file, node_class_labels_file, vertex_dict_file, test_percentage)	
