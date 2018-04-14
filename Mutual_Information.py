from __future__ import division
from decimal import Decimal, localcontext
import operator as op
from math import log
import itertools
from Graph_Operations import *

def mutualInformation(type,targetGraph,predictorGraph):

    if type == "normal":
        return normalMutualInformation(targetGraph)
    elif type == "core":
        return coreMutualInformation(targetGraph,predictorGraph)
    else:
        return globalMutualInformation(targetGraph,predictorGraph)


def normalMutualInformation(graph):

    mutual_info = 0

    M = graph.ecount()
    print(M)

    counter = 0
    for edge in graph.es:

        #if counter % 100 == 0:
        #    print(counter)

        source_index = edge.source
        target_index = edge.target

        #print(graph.vs[source_index]["name"] + " " + graph.vs[target_index]["name"])

        k_m = graph.degree(source_index,mode='ALL')
        k_n = graph.degree(target_index,mode='ALL')

        # MI = Sigma_(z in O_xy)  I(L_xy = 1 ; z) - I(L_xy = 1)

        # First element
        # I(L_mn = 1 | z)
        source_neighbors = graph.neighbors(source_index)
        target_neighbors = graph.neighbors(target_index)

        sharedNeighbors = list(set(source_neighbors).intersection(target_neighbors))

        sigmaMutuals = 0
        for z in sharedNeighbors:
            zNeighbors = graph.neighbors(z)
            try:
                zNeighbors.remove(source_index)
                zNeighbors.remove(target_index)
            except ValueError:
                continue
            if len(zNeighbors) < 2:
                continue
            else:
                #print("in else")
                connectedNodes = 0
                notConnectedNodes = 0
                first_term = 1 / (len(zNeighbors) * (len(zNeighbors) - 1))
                sigmaZ = 0
                pairs = list(itertools.combinations(zNeighbors,2))
                for pair in pairs:
                    k_x = graph.degree(pair[0],mode='ALL')
                    k_y = graph.degree(pair[1],mode='ALL')
                    with localcontext() as cont:
                        cont.prec=100
                        sigmaZ += log(Decimal(ncr(M,k_x))/Decimal(ncr(M,k_x)-ncr(M-k_y,k_x)))
                    if graph.are_connected(pair[0], pair[1]):
                        connectedNodes += 1
                    else:
                        notConnectedNodes += 1
                #print(connectedNodes)
                #print(notConnectedNodes)
                if connectedNodes != 0:
                    second_term = log(connectedNodes/(connectedNodes+notConnectedNodes))
                else:
                    second_term = 0
                sigmaMutuals += first_term * sigmaZ + second_term
                #print(sigmaMutuals)
        #print(sigmaMutuals)
        # Second element
        # Based on equation we have p(L_mn) = 1 -  [ C(M,k_m)-k_n/ C(M,k_n)]
        #print((ncr(M,k_m) - k_n))
        #print(ncr(M,k_m))

        with localcontext() as cont:
            cont.prec=100
            linkProb = 1 - (Decimal(ncr(M,k_m) - k_n)/Decimal(ncr(M,k_m)))

        '''print(linkProb)'''
        # Mutual Information Score
        if linkProb <= 0E-99:
            stepMutual = sigmaMutuals
        else:
            stepMutual = sigmaMutuals - log(linkProb)
        #stepMutual = Decimal(sigmaMutuals) + Decimal(linkProb)
        #print(stepMutual)
        mutual_info += stepMutual

        #print("========================")
        counter += 1
    return mutual_info,M

def coreMutualInformation(graph1,graph2):

    mutual_info = 0

    M = graph1.ecount()
    print(M)

    counter = 0
    for edge in graph1.es:

        #if counter % 100 == 0:
        #    print(counter)

        source_index = edge.source
        target_index = edge.target

        #print(graph.vs[source_index]["name"] + " " + graph.vs[target_index]["name"])

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # For core negative relations we need 2 and 4
        # For core positive relations we don't need 2 and 4
        #k_m = graph1.degree(source_index,mode='ALL')
        #k_m += graph2.degree(source_index,mode='ALL')
        #k_n = graph1.degree(target_index,mode='ALL')
        #k_n += graph2.degree(target_index,mode='ALL')

        # MI = Sigma_(z in O_xy)  I(L_xy = 1 ; z) - I(L_xy = 1)

        # First element
        # I(L_mn = 1 | z)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # For now I just change this to core neighbors
        source_neighbors_1 = graph1.neighbors(source_index)
        list1 = findNames(graph1,source_neighbors_1)
        source_neighbors_2 = graph2.neighbors(source_index)
        list2 = findNames(graph2,source_neighbors_2)

        source_neighbors = findIndices(graph1,coreNeighbors(list1,list2))
        k_m = len(coreNeighbors(list1,list2))

        target_neighbors_1 = graph1.neighbors(target_index)
        list1 = findNames(graph1,target_neighbors_1)
        target_neighbors_2 = graph2.neighbors(target_index)
        list2 = findNames(graph2,target_neighbors_2)

        target_neighbors = findIndices(graph1,coreNeighbors(list1,list2))
        k_n = len(coreNeighbors(list1,list2))

        sharedNeighbors = list(set(source_neighbors).intersection(target_neighbors))

        sigmaMutuals = 0
        for z in sharedNeighbors:
            zNeighbors = graph1.neighbors(z)
            try:
                zNeighbors.remove(source_index)
                zNeighbors.remove(target_index)
            except ValueError:
                continue
            if len(zNeighbors) < 2:
                continue
            else:
                #print("in else")
                connectedNodes = 0
                notConnectedNodes = 0
                first_term = 1 / (len(zNeighbors) * (len(zNeighbors) - 1))
                sigmaZ = 0
                pairs = list(itertools.combinations(zNeighbors,2))
                for pair in pairs:
                    k_x = graph1.degree(pair[0],mode='ALL')
                    k_y = graph1.degree(pair[1],mode='ALL')
                    with localcontext() as cont:
                        cont.prec=100
                        sigmaZ += log(Decimal(ncr(M,k_x))/Decimal(ncr(M,k_x)-ncr(M-k_y,k_x)))
                    if graph1.are_connected(pair[0], pair[1]):
                        connectedNodes += 1
                    else:
                        notConnectedNodes += 1
                #print(connectedNodes)
                #print(notConnectedNodes)
                if connectedNodes != 0:
                    second_term = log(connectedNodes/(connectedNodes+notConnectedNodes))
                else:
                    second_term = 0
                sigmaMutuals += first_term * sigmaZ + second_term
                #print(sigmaMutuals)
        #print(sigmaMutuals)
        # Second element
        # Based on equation we have p(L_mn) = 1 -  [ C(M,k_m)-k_n/ C(M,k_n)]
        #print((ncr(M,k_m) - k_n))
        #print(ncr(M,k_m))

        with localcontext() as cont:
            cont.prec=100
            linkProb = 1 - Decimal(ncr(M,k_m) - k_n)/Decimal(ncr(M,k_m))

        # Mutual Information Score
        if linkProb <= 0E-99:
            stepMutual = sigmaMutuals
        else:
            stepMutual = sigmaMutuals - log(linkProb)
        #stepMutual = Decimal(sigmaMutuals) + Decimal(linkProb)
        #print(stepMutual)
        mutual_info += stepMutual

        #print("========================")
        counter += 1
    return mutual_info,M

def globalMutualInformation(graph1,graph2):

    mutual_info = 0

    M = graph1.ecount()
    print(M)

    counter = 0
    for edge in graph1.es:

        #if counter % 100 == 0:
        #    print(counter)

        source_index = edge.source
        target_index = edge.target

        #print(graph.vs[source_index]["name"] + " " + graph.vs[target_index]["name"])

        #k_m = graph1.degree(source_index,mode='ALL')
        #k_m += graph2.degree(source_index,mode='ALL')
        #k_n = graph1.degree(target_index,mode='ALL')
        #k_n += graph2.degree(target_index,mode='ALL')

        # MI = Sigma_(z in O_xy)  I(L_xy = 1 ; z) - I(L_xy = 1)

        # First element
        # I(L_mn = 1 | z)
        source_neighbors_1 = graph1.neighbors(source_index)
        list1 = findNames(graph1,source_neighbors_1)
        source_neighbors_2 = graph2.neighbors(source_index)
        list2 = findNames(graph2,source_neighbors_2)

        source_neighbors = findIndices(graph1,globalNeighbors(list1,list2))
        k_m = len(globalNeighbors(list1,list2))

        target_neighbors_1 = graph1.neighbors(target_index)
        list1 = findNames(graph1,target_neighbors_1)
        target_neighbors_2 = graph2.neighbors(target_index)
        list2 = findNames(graph2,target_neighbors_2)

        target_neighbors = findIndices(graph1,globalNeighbors(list1,list2))
        k_n = len(globalNeighbors(list1,list2))

        sharedNeighbors = list(set(source_neighbors).intersection(target_neighbors))

        sigmaMutuals = 0
        for z in sharedNeighbors:
            zNeighbors = graph1.neighbors(z)
            try:
                zNeighbors.remove(source_index)
                zNeighbors.remove(target_index)
            except ValueError:
                continue
            if len(zNeighbors) < 2:
                continue
            else:
                #print("in else")
                connectedNodes = 0
                notConnectedNodes = 0
                first_term = 1 / (len(zNeighbors) * (len(zNeighbors) - 1))
                sigmaZ = 0
                pairs = list(itertools.combinations(zNeighbors,2))
                for pair in pairs:
                    k_x = graph1.degree(pair[0],mode='ALL')
                    k_y = graph1.degree(pair[1],mode='ALL')
                    with localcontext() as cont:
                        cont.prec=100
                        sigmaZ += log(Decimal(ncr(M,k_x))/Decimal(ncr(M,k_x)-ncr(M-k_y,k_x)))
                    if graph1.are_connected(pair[0], pair[1]):
                        connectedNodes += 1
                    else:
                        notConnectedNodes += 1
                #print(connectedNodes)
                #print(notConnectedNodes)
                if connectedNodes != 0:
                    second_term = log(connectedNodes/(connectedNodes+notConnectedNodes))
                else:
                    second_term = 0
                sigmaMutuals += first_term * sigmaZ + second_term
                #print(sigmaMutuals)
        #print(sigmaMutuals)
        # Second element
        # Based on equation we have p(L_mn) = 1 -  [ C(M,k_m)-k_n/ C(M,k_n)]
        #print((ncr(M,k_m) - k_n))
        #print(ncr(M,k_m))

        with localcontext() as cont:
            cont.prec=100
            linkProb = 1 - Decimal(ncr(M,k_m) - k_n)/Decimal(ncr(M,k_m))

        '''print(linkProb)'''
        # Mutual Information Score
        if linkProb <= 0E-99:
            stepMutual = sigmaMutuals
        else:
            stepMutual = sigmaMutuals - log(linkProb)
        #stepMutual = Decimal(sigmaMutuals) + Decimal(linkProb)
        #print(stepMutual)
        mutual_info += stepMutual

        #print("========================")
        counter += 1
    return mutual_info,M

def ncr(n, r):
    if n >= r:
        r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom