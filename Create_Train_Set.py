# Function implemented to create train sets
import Feature_Handler
import random
from random import randint

def createTrainSet(graph,name,numPosSamples,numNegSamples,activeFeatures,directory,date):

    print("STEP : Train set ("+ name +") for " + str(numPosSamples) +" positive and " + str(numNegSamples) + " negative samples")
    # Create matrix for selected features
    AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,\
    inDegrees,outDegrees,pageRanks,betweenness,dailyRate = Feature_Handler.createFeatureMatrix(graph,activeFeatures,date,name,"train")

    trainFile = open(directory+name+"-train-" + date + ".txt","w")

    # ====================== CREATE NEGATIVE TRAIN SET =======================
    negativeSet = []
    AA = graph.get_adjacency()

    j = 0
    while j < numNegSamples:
        x = randint(0,graph.vcount()-1)
        y = randint(0,graph.vcount()-1)
        if AA[x][y] == 0:
            AA[x][y] = -1
            negativeSet.append([x,y])
            j+=1

    for s in negativeSet:
        node1 = s[0];
        node2 = s[1];
        feature_vector = Feature_Handler.create_feature_vector\
            (node1,node2,AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,
             inDegrees,outDegrees,pageRanks,betweenness,dailyRate,activeFeatures)
        trainFile.write("0" + feature_vector+"\n")
        #trainFile.write(feature_vector+"0\n")

    # ====================== CREATE POSITIVE TRAIN SET =======================
    posRandomIndex = random.sample(range(0,graph.ecount()), numPosSamples)
    #print(posRandomIndex)
    for i in posRandomIndex:
        node1 = graph.es[i].tuple[0]
        node2 = graph.es[i].tuple[1]
        feature_vector = Feature_Handler.create_feature_vector\
            (node1,node2,AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,
             inDegrees,outDegrees,pageRanks,betweenness,dailyRate,activeFeatures)
        overSample = 0
        while overSample < 1:
            trainFile.write("1" + feature_vector+"\n")
            #trainFile.write(feature_vector+"1\n")
            overSample += 1

    trainFile.close()

    return name+"-train-" + date + ".txt"