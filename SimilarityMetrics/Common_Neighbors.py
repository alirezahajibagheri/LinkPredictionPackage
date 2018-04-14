from itertools import combinations
from Calculate_Unsupervised_AUROC import calculate_auroc
import numpy as np
from Graph_Operations import *

def common_neighbors_score(graph):
    n = graph.vcount()
    common_neis = np.zeros((n, n))
    for v in range(graph.vcount()):
        neis = graph.neighbors(v)
        for u, w in combinations(neis, 2):
            # v is a common neighbor of u and w
            common_neis[u, w] += 1
            common_neis[w, u] += 1
    for v in range(graph.vcount()):
        for w in range(graph.vcount()):
            if v == w :
                common_neis[v][w] = 0
    return common_neis

# Layer1 trades
# Layer2 messages
def multilayer_common_neighbors_score(layer1,layer2,type):

    n = layer1.vcount()
    common_neis = np.zeros((n, n))

    for v in range(layer1.vcount()):
        for w in range(layer1.vcount()):
            if v == w:
                continue
            neighbors1 = findNames(layer1,layer1.neighbors(v))
            neighbors2 = findNames(layer1,layer1.neighbors(w))
            common_neis[v,w] = len(set(neighbors1).intersection(neighbors2))
    cn_layer2 = common_neighbors_score(layer2)
    for v in range(layer1.vcount()):
        for w in range(layer1.vcount()):
            node1 = layer1.vs[v]["name"]
            node2 = layer1.vs[w]["name"]
            try:
                index1 = layer2.vs.find(name=node1).index
                index2 = layer2.vs.find(name=node2).index
            except ValueError:
                continue

            if type == "core":
                neis1_layer1 = findNames(layer1,layer1.neighbors(v))
                neis1_layer2 = findNames(layer2,layer2.neighbors(index1))
                neis2_layer1 = findNames(layer1,layer1.neighbors(w))
                neis2_layer2 = findNames(layer2,layer2.neighbors(index2))
                neighbors1 = coreNeighbors(neis1_layer1,neis1_layer2)
                neighbors2 = coreNeighbors(neis2_layer1,neis2_layer2)
                common_neis[v,w] = len(set(neighbors1).intersection(neighbors2))
            elif type == "global":
                common_neis[v][w] = cn_layer2[index1][index2] + common_neis[v][w]
    return common_neis


def common_neighbors_prediction(trainGraph,testGraph,mode,layer2,type):

    adjacencyMatrix = testGraph.get_adjacency()
    if mode == "multilayer":
        CNMatrix = multilayer_common_neighbors_score(trainGraph,layer2,type)
    if mode == "normal":
        CNMatrix = common_neighbors_score(trainGraph)

    CNMatrix  = np.array(CNMatrix)
    i = (-CNMatrix ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, CNMatrix .shape)
    sortedIndices = np.vstack(j).T

    sortedList = np.sort(np.array(CNMatrix).ravel())
    roc_auc = calculate_auroc(sortedList[::-1],sortedIndices,adjacencyMatrix)

    #roc_auc = auc(fpr, tpr)
    #print ("Area under the ROC curve : %f" % roc_auc)

    #return roc_auc,fpr,tpr
    #return roc_auc,precision,recall
    return roc_auc,sortedIndices
