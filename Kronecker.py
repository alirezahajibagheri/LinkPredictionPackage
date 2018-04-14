import networkx as nx
import random

def weightedRandom(weights, numChoices = 1):
    import bisect
    import numpy as np
    choices = range(len(weights))
    cumdist = np.cumsum(weights)
    if numChoices == 1:
        x = random.random() * cumdist[-1]
        return choices[bisect.bisect(cumdist, x)]
    else:
        results = []
        for _ in range(numChoices):
            x = random.random() * cumdist[-1]
            results.append(choices[bisect.bisect(cumdist, x)])

def edgeProb(i, j, init, iterations):
    """Helper function to determine the probability of an edge appearing here"""
    prob = 1
    for k in range(iterations):
        prob *= init[i % init.shape[0], j % init.shape[0]]
        if prob == 0:
            return 0
        i = int(i/init.shape[0])
        j = int(j/init.shape[0])
    return prob

def genKronecker(init, iterations):
    """Generates a stochastic Kronecker graph from starting matrix init with the given number of iterations"""
    numNodes = init.shape[0]**iterations
    kg = nx.Graph()    
    kg.add_nodes_from(range(numNodes))
    for i in range(numNodes):
        for j in range(numNodes):
            if random.random() < edgeProb(i, j, init, iterations):
                kg.add_edge(i,j)
    return kg

def genFastKronecker(init, iterations, numEdges = None):
    """Generate a Kronecker graph quickly by recursive descent"""
    if numEdges == None:
        numEdges = int(init.sum()**iterations)
    #if numNodes == None:
    numNodes = init.shape[0]**iterations
    kg = nx.Graph()
    kg.add_nodes_from(range(numNodes))
    probVector = init.flatten()
    e = 0
    while e < numEdges:
        rng = numNodes
        u = 0
        v = 0
        for k in range(iterations):
            choice = weightedRandom(probVector)
            r = int(choice/init.shape[0])
            c = choice % init.shape[0]
            rng /= init.shape[0]
            u += r*rng
            v += c*rng
        if u != v and not kg.has_edge(u, v):
            kg.add_edge(u,v)
            e += 1
    #print(kg.nodes())
    return kg
            
    
    