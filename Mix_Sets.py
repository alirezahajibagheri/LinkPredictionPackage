# Function implemented to mix train and test sets
import sys
import csv


def mix_sets(set1,set2,n):
    print("STEP : Mix sets")
    firstSet = open("Networks/"+set1,"a")
    secondSet = open("Networks/"+set2,"r")

    i=0
    for line in secondSet:
        # Skip the first lines related to arrf header - n is number of features
        i+=1
        if i < (5+n):
            continue
        else:
            firstSet.write(line)

    firstSet.close()
    return set1

