from __future__ import division
import numpy as np
from Calculate_Unsupervised_AUROC import calculate_auroc
from sklearn.metrics import auc
from math import log
from Graph_Operations import *

def jaccard_coefficient_score(graph):
    """Computes Adar-Adamic similarity matrix for an adjacency matrix"""

    N = graph.vcount()
    JC = np.zeros((N,N))

    JC = graph.similarity_jaccard(vertices=None, pairs=None, mode="All")

    return np.array(JC)

# Layer1 trades
# Layer2 messages
def multilayer_jaccard_coefficient_score(layer1,layer2,type):
    """Computes Adar-Adamic similarity matrix for an adjacency matrix"""

    N = layer1.vcount()
    JC = np.zeros((N,N))

    for v in range(layer1.vcount()):
        for w in range(v+1,layer1.vcount()):
            #print(str(v) + " " + str(w))
            name1 = layer1.vs[v]["name"]
            name2 = layer1.vs[w]["name"]
            try:
                index1 = layer2.vs.find(name=name1).index
                index2 = layer2.vs.find(name=name2).index
            except IndexError:
                continue
            except ValueError:
                continue
            neis1_layer1 = findNames(layer1,layer1.neighbors(v))
            neis1_layer2 = findNames(layer2,layer2.neighbors(index1))
            neis2_layer1 = findNames(layer1,layer1.neighbors(w))
            neis2_layer2 = findNames(layer2,layer2.neighbors(index2))
            if type == "core":
                neighbors1 = coreNeighbors(neis1_layer1,neis1_layer2)
                neighbors2 = coreNeighbors(neis2_layer1,neis2_layer2)
            elif type == "global":
                neighbors1 = globalNeighbors(neis1_layer1,neis1_layer2)
                neighbors2 = globalNeighbors(neis2_layer1,neis2_layer2)

            if len(set(neighbors1).union(neighbors2)) != 0:
                JC[v,w] = len(set(neighbors1).intersection(neighbors2))/len(set(neighbors1).union(neighbors2))
                JC[w,v] = JC[v,w]

    return np.array(JC)

def jaccard_coefficient_prediction(trainGraph,testGraph,mode,layer2,type):

    adjacencyMatrix = testGraph.get_adjacency()
    if mode == "multilayer":
        JCMatrix = multilayer_jaccard_coefficient_score(trainGraph,layer2,type)
    if mode == "normal":
        JCMatrix = jaccard_coefficient_score(trainGraph)
    JCMatrix  = np.array(JCMatrix)
    i = (-JCMatrix ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, JCMatrix .shape)
    sortedIndices = np.vstack(j).T
    total = len(JCMatrix)*len(JCMatrix)

    sortedList = np.sort(np.array(JCMatrix).ravel())

    roc_auc = calculate_auroc(sortedList[::-1],sortedIndices,adjacencyMatrix)

    #roc_auc = auc(fpr, tpr)
    #print ("Area under the ROC curve : %f" % roc_auc)

    #return roc_auc,fpr,tpr
    #return roc_auc,precision,recall
    return roc_auc,sortedIndices

# ====================== OLD CODE =======================
'''for v in range(layer1.vcount()):
        list1 = layer1.neighbors(v)
        try:
            list2 = layer2.neighbors(layer2.vs.find(name=layer1.vs[v]["name"]).index)
        except ValueError:
            list2 = []
        list1 = findNames(layer1,list1)
        list2 = findNames(layer2,list2)
        neighb1 = list(set(list1 + list2))
        for w in range(layer1.vcount()):
            list11 = layer1.neighbors(w)
            try:
                list22 = layer2.neighbors(layer2.vs.find(name=layer1.vs[w]["name"]).index)
            except ValueError:
                list22 = []
            list11 = findNames(layer1,list11)
            list22 = findNames(layer2,list22)
            neighb2 = list(set(list11 + list22))
            try:
                JC[v][w] = len(list(set(neighb1).intersection(neighb2)))/len(neighb1+neighb2)
            except ZeroDivisionError:
                JC[v][w] = 0


    JC = layer1.similarity_jaccard(vertices=None, pairs=None, mode="All")

    JC_layer2 = jaccard_coefficient_score(layer2)
    for v in range(layer1.vcount()):
        for w in range(layer1.vcount()):
            node1 = layer1.vs[v]["name"]
            node2 = layer1.vs[w]["name"]
            try:
                index1 = layer2.vs.find(name=node1).index
                index2 = layer2.vs.find(name=node2).index
                if JC_layer2[index1][index2] != 0 and JC[v][w]!= 0:
                    JC[v][w] = JC_layer2[index1][index2] * JC[v][w]
                if JC_layer2[index1][index2] != 0 and JC[v][w] == 0:
                    JC[v][w] = log(JC_layer2[index1][index2],2)
            except ValueError:
                continue
    '''