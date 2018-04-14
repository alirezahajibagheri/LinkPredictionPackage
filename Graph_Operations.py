from __future__ import division
import numpy as np
from math import log
from igraph import *
from Kronecker import *

# We want to add X% new edges from g2 to g1
# Receives two networkx graphs
# Add edges to first one until it has X% similarity with the second one
def transformGraph(g1,g2,alpha):

    e1 = g1.edges()
    e2 = g2.edges()
    sharedEdges = []

    for x in e1:
        if x in e2:
            sharedEdges.append(x)

    g1.remove_edges_from(sharedEdges)
    sample_size = int(math.ceil(alpha * g2.number_of_edges()))
    sample = random.sample(e2, sample_size)
    g1.add_edges_from(sample)

    return g1

# Create a set of edges different in two graphs
def graphDifference(g1,g2):
    diffSet = []

    for edge in g2.edges():
        if edge not in g1.edges():
            diffSet.append(edge)
    return diffSet;

def coreNeighbors(neighb1,neighb2):

    CN = list(set(neighb1).intersection(neighb2))
    return CN

def globalNeighbors(neighb1,neighb2):

    GN = list(set(neighb1 + neighb2))
    return GN

# Create full graphs from two graphs g1 and g2
# The retuned graph will have nodes from both
def fullGraph(g1,g2,mode):

    full = Graph()

    temp = nx.Graph()
    temp.add_nodes_from(g1)
    temp.add_nodes_from(g2)
    temp.add_edges_from(g1.edges())
    temp.add_edges_from(g2.edges())
    diffSet = graphDifference(g1,g2)
    temp.remove_edges_from(diffSet)

    if mode == "igraph":
        full.add_vertices(temp.nodes())
        full.add_edges(temp.edges())
        return full
    if mode == "networkx":
        return temp

# Convert networkx to igraph
def convertToiGraph(graph):

    newGraph = Graph()
    newGraph.add_vertices(graph.nodes())
    newGraph.add_edges(graph.edges())
    return newGraph

def findNames(graph,indexList):

    names = []

    for v in indexList:
        names.append(graph.vs[v]["name"])

    return names

def findIndices(graph,nameList):

    indices = []

    for v in nameList:
        indices.append(graph.vs[v]["name"])

    return indices