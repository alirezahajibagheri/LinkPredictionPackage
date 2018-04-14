# Author: Alireza Hajibagheri
# Positive Link Prediction in Dynamic Heterogeneous Networks
# Main file of the project
# Name of the files are Trades and Messages, however those two
# could be any arbitrary pair of layers from different networks.
#============================================================
from __future__ import division
import sys
sys.path.insert(0, 'SimilarityMetrics')
from igraph import *
import Common_Neighbors
import Preferential_Attachment
import Jaccard_Coefficient
import Adamic_Adar
from Rank_Aggregation import *
from Configurations import directory_unsupervised
from Calculate_Unsupervised_AUROC import calculate_auroc



def layer_weighting(targetLayer,otherLayers):

    weights = []
    total = targetLayer.ecount()
    for layer in otherLayers:
        tempAdj = np.array(layer.get_adjacency().data)
        similarity = 0
        for pair in targetLayer.es:
            nodes = pair.tuple
            node1 = targetLayer.vs[nodes[0]]["name"]
            node2 = targetLayer.vs[nodes[1]]["name"]
            try:
                index1 = layer.vs.find(name=node1).index
                index2 = layer.vs.find(name=node2).index
                if tempAdj[index1][index2] == 1:
                    similarity += 1
            except ValueError:
                continue
        weights.append(similarity/total)

    return weights


def graph_weighting(graph,layerWeights):

    return graph

def memScoreMatch(members,scores):

    # List of tuples (edges)
    #members = list(members.keys())
    # Attach tuples to their score so we can sort them together
    yx = zip(scores, members)
    # Sort attached tuples,scores
    yx.sort(reverse=True)
    # List of sorted edges
    members_sorted = [x for y, x in yx]
    # Convert list of tuples to numpy array for othe functions
    members_sorted = np.asarray(members_sorted)#[list(elem) for elem in kk]
    # List of sorted scores
    scores_sorted = [y for y, x in yx]

    return members_sorted,np.array(scores_sorted)

    return 0

# PARAMETERS (DATE, ...)
#===========================================================
Aggregated = []
directory = directory_unsupervised
tradeNodeRates = defaultdict(int)
messageNodeRates = defaultdict(int)

#===========================================================
for ii in range(1,31):
    print("ii is : " + str(ii))
    jj = ii + 1
    start = str(ii)
    end = str(jj)
    if ii < 10:
        start = "0" + start
    if jj < 10:
        end = "0" + end
    trainFromDate = "2009-12-" + start
    trainToDate = "2009-12-" + start
    testFromDate = "2009-12-" + end
    testToDate = "2009-12-" + end

    # Create train network(s) csv files from MySQL database
    # If you have csv files available, replace function with the
    # file name and place it in a directory called "Networks" in
    # the same folder as the code is.
    #============================================================
    tradeTrainFile = "name_of_the_layer_file" + trainFromDate + ".txt"
    messageTrainFile = "name_of_the_layer_file" + trainFromDate + ".txt"
    tradeTestFile = "name_of_the_layer_file" + testFromDate + ".txt"
    messageTestFile = "name_of_the_layer_file" + testFromDate + ".txt"

    #================= Layer1 =================
    tradesFullFile = "name_of_the_full_network_file"
    tradesFullNet = open(directory+tradesFullFile,'r')
    tradesFullGraph = Graph.Read_Ncol(tradesFullNet, names=True, weights="if_present", directed=True)
    trainFile = open(directory+tradeTrainFile,'r')
    testFile = open(directory+tradeTestFile,'r')
    tradesTrainGraph = tradesFullGraph.copy()
    tradesTestGraph = tradesFullGraph.copy()

    for line in testFile:
        node1 = line.split(" ")[0]
        node2 = line.split(" ")[1].rstrip()
        i = tradesTrainGraph.vs.find(name=node1).index
        j = tradesTrainGraph.vs.find(name=node2).index
        tradesTrainGraph.delete_edges([(i,j)])

    tradeTempRate = defaultdict(int)
    for line in trainFile:
        node1 = line.split(" ")[0]
        node2 = line.split(" ")[1].rstrip()
        i = tradesTestGraph.vs.find(name=node1).index
        j = tradesTestGraph.vs.find(name=node2).index
        tradesTestGraph.delete_edges([(i,j)])
        # Add node rate
        tradeTempRate[i] += 1
        if ii == 1:
            tradeNodeRates[i] = 1

    #================= Layer2 =================
    messagesFullFile = "name_of_the_full_network_file"
    messagesFullNet = open(directory+messagesFullFile,'r')
    messagesFullGraph = Graph.Read_Ncol(messagesFullNet, names=True, weights="if_present", directed=True)
    trainFile = open(directory+messageTrainFile,'r')
    testFile = open(directory+messageTestFile,'r')
    messagesTrainGraph = messagesFullGraph.copy()
    messagesTestGraph = messagesFullGraph.copy()

    for line in testFile:
        node1 = line.split(" ")[0]
        node2 = line.split(" ")[1].rstrip()
        i = messagesTrainGraph.vs.find(name=node1).index
        j = messagesTrainGraph.vs.find(name=node2).index
        messagesTrainGraph.delete_edges([(i,j)])

    messageTempRate = defaultdict(int)
    for line in trainFile:
        node1 = line.split(" ")[0]
        node2 = line.split(" ")[1].rstrip()
        i = messagesTestGraph.vs.find(name=node1).index
        j = messagesTestGraph.vs.find(name=node2).index
        messagesTestGraph.delete_edges([(i,j)])
        # Add node rate
        messageTempRate[i] += 1
        if ii == 1:
            messageNodeRates[i] = 1

    tradeLayerWeights = layer_weighting(tradesTrainGraph,[messagesTrainGraph])
    messageLayerWeights = layer_weighting(messagesTrainGraph,[tradesTrainGraph])

    for key in messageTempRate:
        if key in messageNodeRates:
            messageNodeRates[key] = messageNodeRates.get(key)+messageTempRate.get(key)/2

    for key in tradeTempRate:
        if key in tradeNodeRates:
            tradeNodeRates[key] = tradeNodeRates.get(key)+tradeTempRate.get(key)/2

    pairs = []
    scores = []

    N = tradesTrainGraph.vcount()
    for v in range(N):
        for w in range(N):
            pairs.append((v,w))
            if v in tradeNodeRates:
                scores.append(tradeNodeRates.get(v) + (tradeLayerWeights[0] * 1))
            else:
                scores.append(0)

    sortedIndices,sortedList = memScoreMatch(pairs,scores)
    adjacencyMatrix = tradesTrainGraph.get_adjacency()
    auroc = calculate_auroc(sortedList,sortedIndices,adjacencyMatrix)
    print(auroc)

    scores = []
    pairs = []
    N = messagesTrainGraph.vcount()
    for v in range(N):
        for w in range(N):
            pairs.append((v,w))
            if v in messageNodeRates:
                scores.append(messageNodeRates.get(v) + (messageLayerWeights[0] * 1))
            else:
                scores.append(0)

    sortedIndices,sortedList = memScoreMatch(pairs,scores)
    adjacencyMatrix = messagesTrainGraph.get_adjacency()
    auroc = calculate_auroc(sortedList,sortedIndices,adjacencyMatrix)
    print(auroc)