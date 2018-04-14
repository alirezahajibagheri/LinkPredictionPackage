import networkx as netx
import numpy as np

def betweenness_centrality_score(graph):

    BC = np.zeros((graph.ecount(),graph.ecount()))

    A = [edge.tuple for edge in graph.es]
    G = netx.DiGraph(A)
    betweeness = netx.algorithms.edge_betweenness_centrality(G)
    for key in betweeness.keys():
        BC[key[0]][key[1]] = betweeness[key]

    return BC
