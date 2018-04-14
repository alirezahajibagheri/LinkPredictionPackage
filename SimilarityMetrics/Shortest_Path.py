import numpy as np

def shortest_path(graph):
    """Computes Adar-Adamic similarity matrix for an adjacency matrix"""

    N = graph.vcount()
    SP = np.zeros((N,N))

    SP = graph.shortest_paths_dijkstra(source=None, target=None, weights=None, mode='ALL')

    return SP
