from Calculate_Unsupervised_AUROC import calculate_auroc
from Graph_Operations import *

def adamic_adar_score(graph):
    """Computes Adar-Adamic similarity matrix for an adjacency matrix"""

    N = graph.vcount()
    AA = np.zeros((N,N))

    AA = graph.similarity_inverse_log_weighted(vertices=None, mode="ALL")

    return np.array(AA)

# Layer1 trades
# Layer2 messages
def multilayer_adamic_adar_score(layer1,layer2,type):
    """Computes Adar-Adamic similarity matrix for an adjacency matrix"""

    N = layer1.vcount()
    AA = np.zeros((N,N))
    #AA = layer1.similarity_inverse_log_weighted(vertices=None, mode="ALL")
    for v in range(layer1.vcount()):
        list1 = layer1.neighbors(v)
        try:
            list2 = layer2.neighbors(layer2.vs.find(name=layer1.vs[v]["name"]).index)
        except ValueError:
            list2 = []
        list1 = findNames(layer1,list1)
        list2 = findNames(layer2,list2)
        for w in range(layer1.vcount()):
            list11 = layer1.neighbors(w)
            try:
                list22 = layer2.neighbors(layer2.vs.find(name=layer1.vs[w]["name"]).index)
            except ValueError:
                list22 = []
            list11 = findNames(layer1,list11)
            list22 = findNames(layer2,list22)

            if type == "core":
                neighb1 = list(coreNeighbors(list1,list2))
                neighb2 = list(coreNeighbors(list11,list22))
            elif type == "global":
                neighb1 = list(globalNeighbors(list1,list2))
                neighb2 = list(globalNeighbors(list11,list22))
            sharedNeighbors = list(set(neighb1).intersection(neighb2))
            aaScore = 0
            #print(sharedNeighbors)
            for node in sharedNeighbors:
                try:
                    l1 = layer1.neighbors(layer1.vs.find(name=node).index)
                except ValueError:
                    l1 = []
                try:
                    l2 = layer2.neighbors(layer2.vs.find(name=node).index)
                except ValueError:
                    l2 = []
                l1 = findNames(layer1,l1)
                l2 = findNames(layer2,l2)
                if type == "core":
                    sizeSet = len(coreNeighbors(l1,l2))
                elif type == "global":
                    sizeSet = len(globalNeighbors(l1,l2))

                if sizeSet > 1:
                    aaScore += 1 / log(len(list(set(l1 + l2))))
                else:
                    continue
            AA[v][w] = aaScore


    '''AA_layer2 = adamic_adar_score(layer2)
    for v in range(layer1.vcount()):
        for w in range(layer1.vcount()):
            node1 = layer1.vs[v]["name"]
            node2 = layer1.vs[w]["name"]
            try:
                index1 = layer2.vs.find(name=node1).index
                index2 = layer2.vs.find(name=node2).index
                if AA_layer2[index1][index2] != 0 and AA[v][w]!= 0:
                    AA[v][w] = AA_layer2[index1][index2] * AA[v][w]
                if AA_layer2[index1][index2] != 0 and AA[v][w] == 0:
                    AA[v][w] = log(AA_layer2[index1][index2],2)
            except ValueError:
                continue'''
    return AA

def adamic_adar_prediction(trainGraph,testGraph,mode,layer2,type):

    adjacencyMatrix = testGraph.get_adjacency()
    if mode == "multilayer":
        AAMatrix = multilayer_adamic_adar_score(trainGraph,layer2,type)
    if mode == "normal":
        AAMatrix = adamic_adar_score(trainGraph)
    AAMatrix  = np.array(AAMatrix)
    i = (-AAMatrix ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, AAMatrix .shape)
    sortedIndices = np.vstack(j).T

    total = len(AAMatrix)*len(AAMatrix)

    sortedList = np.sort(np.array(AAMatrix).ravel())

    roc_auc = calculate_auroc(sortedList[::-1],sortedIndices,adjacencyMatrix)
    #roc_auc = auc(fpr, tpr)
    #print ("Area under the ROC curve : %f" % roc_auc)

    #return roc_auc,fpr,tpr
    #return roc_auc,precision,recall
    return roc_auc,sortedIndices

