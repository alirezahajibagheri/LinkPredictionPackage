# Author: Alireza Hajibagheri
# Signed Link Prediction in Dynamic MMOG Networks
# Main file of the project SMLP
#======================================================================
# This returns an average mutual information for layer pairs
# The input is a list of layer names of a network
# Based on the difference between normal mutual information
# and core (global) mutual information values, sign of correlation
# between layers is detected.
#======================================================================
from igraph import *
from sklearn import preprocessing
from Mutual_Information import mutualInformation

# PARAMETERS (DATE, ...)
#===========================================================
# Directory where network information could be found
directory = "Networks/"
# Number of snapshots for the dataset
num_of_snapshots = 31
# List of network layers
layersList = ["messages","raids"]

# Calculating average mutual information for layers
#===========================================================
for targetLayer in layersList:
    for predictorLayer in layersList:
        if targetLayer == predictorLayer:
            continue
        else:
            print("Target : " + targetLayer + " Predictor : " + predictorLayer)

            # File to write mutual information results
            results = open("Outputs/" + targetLayer + "-MI.txt","a")

            # Number of snapshots for the dataset
            for ii in range(1,num_of_snapshots):
                print("ii is : " + str(ii))
                jj = ii + 1
                start = str(ii)
                end = str(jj)
                if ii < 10:
                    start = "0" + start
                if jj < 10:
                    end = "0" + end

                # Read network data from files
                # Reformat the name of the file based on your networks
                #============================================================
                target = targetLayer + "-weighted-2009-12-" + start + ".txt"
                predictor = predictorLayer + "-weighted-2009-12-" + start + ".txt"

                # Create graphs for different layers
                #============================================================
                print("STEP : Creating graph for " + targetLayer)
                targetFile = open(directory+target,'r')
                targetGraph = Graph.Read_Ncol(targetFile, names=True, weights="true", directed=True)
                print("STEP : Creating graph for " + predictorLayer)
                predictorFile = open(directory+predictor,'r')
                predictorGraph = Graph.Read_Ncol(predictorFile, names=True, weights="true", directed=True)

                # MUTUAL INFORMATION
                #============================================================
                # Calculate MI for regular neighborhood set
                normal_mutual_info,num_edges = mutualInformation("normal",targetGraph,predictorGraph)
                # Calculate MI for core neighborhood set
                core_mutual_info,num_edges = mutualInformation("core",targetGraph,predictorGraph)
                # Calculate MI for global neighborhood set
                global_mutual_info,num_edges = mutualInformation("global",targetGraph,predictorGraph)

                # Results are being written into files to help us figure the
                # sign of correlation between layers
                # If core < normal the sign is negative
                # If global > normal the sign is positive
                #============================================================
                #print(normal_mutual_info)
                results.write(str(ii) + ", normal , " + targetLayer + "," + predictorLayer + "," + str(num_edges) + "," + str(normal_mutual_info) + "\n")
                #print(core_mutual_info)
                results.write(str(ii) + ", core , " + targetLayer + "," + predictorLayer + "," + str(num_edges) + "," + str(core_mutual_info) + "\n")
                #print(global_mutual_info)
                results.write(str(ii) + ", global , " + targetLayer + "," + predictorLayer + "," + str(num_edges) + "," + str(global_mutual_info) + "\n")

            results.close()