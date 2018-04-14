import sys
sys.path.insert(0, 'SimilarityMetrics')
import Adamic_Adar
import Jaccard_Coefficient
import Common_Neighbors
import  Preferential_Attachment
import Shortest_Path
import Betweenness_Centrality
from Link_Formation_Analysis import *
from Configurations import time_series
from igraph import *

def createFeatureMatrix(graph,activeFeatures,date,name,mode):

    # Calculate Adamic-Adar matrix
    AAMatrix = []
    if activeFeatures[0]==1:
        AAMatrix = Adamic_Adar.adamic_adar_score(graph)

    # Calculate Jaccard Coefficient matrix
    JCMatrix = []
    if activeFeatures[1]==1:
        JCMatrix = Jaccard_Coefficient.jaccard_coefficient_score(graph)

    # Calculate Common Neighbors matrix
    CNMatrix = []
    if activeFeatures[2]==1:
        CNMatrix = Common_Neighbors.common_neighbors_score(graph)

    # Calculate Preferential Attachment matrix
    PAMatrix = []
    if activeFeatures[3]==1:
        PAMatrix = Preferential_Attachment.preferential_attachment_score(graph)

    # Calculate Shortest Path matrix
    SPMatrix = []
    if activeFeatures[4]==1:
        SPMatrix = Shortest_Path.shortest_path(graph)


    # Calculate in-degree for all nodes
    inDegrees = graph.degree(graph.vs,mode='IN')

    # Calculate out-degree for all nodes
    outDegrees = graph.degree(graph.vs,mode='OUT')

    # Calculate PageRank for all nodes
    pageRanks = graph.pagerank(vertices=None, directed=True, damping=0.85, weights=None, arpack_options=None, implementation='prpack', niter=1000, eps=0.001)

    # Calculate Betweenness Centrality
    betweenness = []
    if activeFeatures[11]==1:
        betweenness = Betweenness_Centrality.betweenness_centrality_score(graph)

    # Time Series Daily Rate Forecasting
    dailyRate = []
    if activeFeatures[12]==1:
        if time_series == "average":
            dailyRate = averageTimeSeriesRate_Enron(graph,date,name,mode)
        if time_series == "maverage":
            dailyRate = movingAverageTimeSeriesRate_Enron(graph,date,name,mode)
        if time_series == "wmaverage":
            dailyRate = wMovingAverageTimeSeriesRate_Enron(graph,date,name,mode)
        if time_series == "smoothing":
            dailyRate = smoothingTimeSeriesRate(graph,date,name,mode)

    return AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,inDegrees,outDegrees,pageRanks,betweenness,dailyRate
    #return AAMatrix,JCMatrix,CNMatrix,PAMatrix,inDegrees,outDegrees,pageRanks,betweenness

def create_header(activeFeatures):

    # Header list for csv
    headerList_csv = ["AA","JC","CN","PA","SP","ID1","ID2","OD1","OD2","PR1","PR2","BC","DR"]
    header_csv = ""
    for i in range(0,len(activeFeatures)):
        if activeFeatures[i] == 1:
            header_csv += headerList_csv[i] + ","
    return header_csv

def create_feature_vector(node1,node2,AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,inDegrees,outDegrees,pageRanks,betweenness,\
                          dailyRate,activeFeatures):
    feature_vector = ""
    # UNSUPERVISED METHODS AS FEATURES
    if activeFeatures[0]==1:
        feature_vector += " " + str(AAMatrix[node1][node2]) #+ ","
    if activeFeatures[1]==1:
        feature_vector += " " + str(JCMatrix[node1][node2]) #+ ","
    if activeFeatures[2]==1:
        feature_vector += " " + str(CNMatrix[node1][node2]) #+ ","
    if activeFeatures[3]==1:
        feature_vector += " " + str(PAMatrix[node1][node2]) #+ ","
    if activeFeatures[4]==1:
        if SPMatrix[node1][node2] == float('inf'):
            SPMatrix[node1][node2]=0
        feature_vector += " " + str(SPMatrix[node1][node2]) #+ ","
    # TOPOLOGICAL FEATURES
    if activeFeatures[5]==1:
        feature_vector += " " + str(inDegrees[node1]) #+ ","
    if activeFeatures[6]==1:
        feature_vector += " " + str(inDegrees[node2]) #+ ","
    if activeFeatures[7]==1:
        feature_vector += " " + str(outDegrees[node1]) #+ ","
    if activeFeatures[8]==1:
        feature_vector += " " + str(outDegrees[node2]) #+ ","
    if activeFeatures[9]==1:
        feature_vector += " " + str(pageRanks[node1]) #+ ","
    if activeFeatures[10]==1:
        feature_vector += " " + str(pageRanks[node2]) #+ ","
    if activeFeatures[11]==1:
        feature_vector += " " + str(betweenness[(node1,node2)]) #+ ","
    # ENGINEERED FEATURES
    if activeFeatures[12]==1:
        feature_vector += " " + str(dailyRate[node1]) #+ ","
    return feature_vector