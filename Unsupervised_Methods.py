# Author: Alireza Hajibagheri
# Positive Link Prediction in Dynamic Heterogeneous Networks
# Main file of the project
#============================================================
import sys
sys.path.insert(0, 'SimilarityMetrics')
import Create_Networks_CSV
from igraph import *
import Common_Neighbors
import Preferential_Attachment
import Jaccard_Coefficient
import Adamic_Adar
from Rank_Aggregation import *
from Configurations import directory_unsupervised
from Calculate_Unsupervised_AUROC import calculate_auroc


# PARAMETERS (DATE, ...)
#===========================================================
AA_AUROC = []
CN_AUROC = []
PA_AUROC = []
JC_AUROC = []
Aggregated = []


directory = directory_unsupervised

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
    #tradeTrainFile = "trades-train-network-2009-12-01.txt"
    tradeTrainFile = Create_Networks_CSV.createNetworkCSV("trades","train",trainFromDate,trainToDate,directory)
    #messageTrainFile = "messages-train-network-2009-12-01.txt"
    messageTrainFile = Create_Networks_CSV.createNetworkCSV("messages","train",trainFromDate,trainToDate,directory)
    #tradeTestFile = "trades-test-network-2009-12-21.txt"
    tradeTestFile = Create_Networks_CSV.createNetworkCSV("trades","test",testFromDate,testToDate,directory)
    #messageTestFile = "messages-test-network-2009-12-21.txt"
    messageTestFile = Create_Networks_CSV.createNetworkCSV("messages","test",testFromDate,testToDate,directory)


    #================= Trades =================
    tradesFullFile = Create_Networks_CSV.createNetworkCSV("trades","full",trainFromDate,testToDate,directory)
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
    for line in trainFile:
        node1 = line.split(" ")[0]
        node2 = line.split(" ")[1].rstrip()
        i = tradesTestGraph.vs.find(name=node1).index
        j = tradesTestGraph.vs.find(name=node2).index
        tradesTestGraph.delete_edges([(i,j)])

    #================= Messages =================
    messagesFullFile = Create_Networks_CSV.createNetworkCSV("messages","full",trainFromDate,testToDate,directory)
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
    for line in trainFile:
        node1 = line.split(" ")[0]
        node2 = line.split(" ")[1].rstrip()
        i = messagesTestGraph.vs.find(name=node1).index
        j = messagesTestGraph.vs.find(name=node2).index
        messagesTestGraph.delete_edges([(i,j)])

    rankedLists = []

    # COMMON NEIGHBORS 1- Normal 2- Core 3- Global
    CN2 = Common_Neighbors.common_neighbors_score(tradesTrainGraph)
    CN1 = Common_Neighbors.common_neighbors_score(messagesTrainGraph)
    for v in range(CN2.shape[0]):
        for w in range(CN2.shape[0]):
            node1 = messagesTrainGraph.vs[v]["name"]
            node2 = messagesTrainGraph.vs[w]["name"]
            try:
                index1 = tradesTrainGraph.vs.find(name=node1).index
                index2 = tradesTrainGraph.vs.find(name=node2).index
                CN2[v][w] = (CN2[v][w] + CN1[index1][index2])/2
            except ValueError:
                continue

    CN2  = np.array(CN2)
    i = (-CN2 ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, CN2 .shape)
    list1 = np.vstack(j).T
    rankedLists.append(list1.tolist())

    AA2 = Adamic_Adar.adamic_adar_score(tradesTrainGraph)
    AA1 = Adamic_Adar.adamic_adar_score(messagesTrainGraph)
    for v in range(AA2.shape[0]):
        for w in range(AA2.shape[0]):
            node1 = messagesTrainGraph.vs[v]["name"]
            node2 = messagesTrainGraph.vs[w]["name"]
            try:
                index1 = tradesTrainGraph.vs.find(name=node1).index
                index2 = tradesTrainGraph.vs.find(name=node2).index
                AA2[v][w] = (AA2[v][w] + AA1[index1][index2])/2
            except ValueError:
                continue

    AA2  = np.array(AA2)
    i = (-AA2 ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, AA2 .shape)
    list2 = np.vstack(j).T
    rankedLists.append(list2.tolist())

    JC2 = Jaccard_Coefficient.jaccard_coefficient_score(tradesTrainGraph)
    JC1 = Jaccard_Coefficient.jaccard_coefficient_score(messagesTrainGraph)
    for v in range(JC2.shape[0]):
        for w in range(JC2.shape[0]):
            node1 = messagesTrainGraph.vs[v]["name"]
            node2 = messagesTrainGraph.vs[w]["name"]
            try:
                index1 = tradesTrainGraph.vs.find(name=node1).index
                index2 = tradesTrainGraph.vs.find(name=node2).index
                JC2[v][w] = (JC2[v][w] + JC1[index1][index2])/2
            except ValueError:
                continue

    JC2  = np.array(JC2)
    i = (-JC2 ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, JC2 .shape)
    list3 = np.vstack(j).T
    rankedLists.append(list3.tolist())

    PA2 = Preferential_Attachment.preferential_attachment_score(tradesTrainGraph)
    PA1 = Preferential_Attachment.preferential_attachment_score(messagesTrainGraph)
    for v in range(PA2.shape[0]):
        for w in range(PA2.shape[0]):
            node1 = messagesTrainGraph.vs[v]["name"]
            node2 = messagesTrainGraph.vs[w]["name"]
            try:
                index1 = tradesTrainGraph.vs.find(name=node1).index
                index2 = tradesTrainGraph.vs.find(name=node2).index
                PA2[v][w] = (PA2[v][w] + PA1[index1][index2])/2
            except ValueError:
                continue

    PA2  = np.array(PA2)
    i = (-PA2 ).argsort(axis=None, kind='mergesort')
    j = np.unravel_index(i, PA2 .shape)
    list4 = np.vstack(j).T
    rankedLists.append(list4.tolist())

    sortedIndices,sortedList = Borda_Rank_Aggregation(rankedLists)
    adjacencyMatrix = tradesTestGraph.get_adjacency()
    auroc = calculate_auroc(sortedList,sortedIndices,adjacencyMatrix)
    Aggregated.append(auroc)
    print(auroc)
    print("=========================================")

print(AA_AUROC)
print(PA_AUROC)
print(JC_AUROC)
print(CN_AUROC)
print(Aggregated)

