import numpy as np
from Calculate_Unsupervised_AUROC import calculate_auroc
from sklearn.metrics import auc
from math import log
from Graph_Operations import *

def preferential_attachment_score(graph):

    A = graph.get_adjacency();
    i_degree = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        i_degree[i] = len(graph.neighbors(i))
        # If we want outdegree instead of total degree:
        #i_degree[i] = graph.vs[i].outdegree()
    PA = np.zeros(A.shape)
    for i in range(PA.shape[0]):
        for j in range(PA.shape[0]):
            PA[i,j] = i_degree[i]*i_degree[j]
    return PA

# Layer1 trades
# Layer2 messages
def multilayer_preferential_attachment_score(layer1,layer2,type):

    A = layer1.get_adjacency();
    PA = np.zeros(A.shape)

    for v in range(layer1.vcount()):
        for w in range(layer1.vcount()):
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
            PA[v,w] = len(neighbors1)*len(neighbors2)
            #PA[w,v] = len(neighbors1)*len(neighbors2)

    return PA

def preferential_attachment_prediction(trainGraph,testGraph,mode,layer2,type):

    adjacencyMatrix = testGraph.get_adjacency()
    if mode == "multilayer":
        PAMatrix = multilayer_preferential_attachment_score(trainGraph,layer2,type)
    if mode == "normal":
        PAMatrix = preferential_attachment_score(trainGraph)
    PAMatrix  = np.array(PAMatrix)
    i = (-PAMatrix ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, PAMatrix .shape)
    sortedIndices = np.vstack(j).T
    total = len(PAMatrix)*len(PAMatrix)

    sortedList = np.sort(np.array(PAMatrix).ravel())

    roc_auc= calculate_auroc(sortedList[::-1],sortedIndices,adjacencyMatrix)

    #roc_auc = auc(fpr, tpr)
    #print ("Area under the ROC curve : %f" % roc_auc)

    #return roc_auc,fpr,tpr
    #return roc_auc,precision,recall
    return roc_auc,sortedIndices

#=================== OLD CODE ========================
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
            PA[v][w] = len(neighb1) * len(neighb2)


    A = layer1.get_adjacency();
    i_degree = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        i_degree[i] = layer1.vs[i].outdegree()
    PA = np.zeros(A.shape)
    for i in range(PA.shape[0]):
        for j in range(PA.shape[0]):
            PA[i,j] = i_degree[i]*i_degree[j]

    PA_layer2 = preferential_attachment_score(layer2)
    for v in range(layer1.vcount()):
        for w in range(layer1.vcount()):
            node1 = layer1.vs[v]["name"]
            node2 = layer1.vs[w]["name"]
            try:
                index1 = layer2.vs.find(name=node1).index
                index2 = layer2.vs.find(name=node2).index
                if PA_layer2[index1][index2] != 0 and PA[v][w]!= 0:
                    PA[v][w] = PA_layer2[index1][index2] * PA[v][w]
                if PA_layer2[index1][index2] != 0 and PA[v][w] == 0:
                    PA[v][w] = log(PA_layer2[index1][index2],2)
            except ValueError:
                continue
    '''