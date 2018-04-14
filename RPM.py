# UNCOMMENT IF YOU NEED SPARK
import sys
import os
from Configurations import osName,directory_supervised,dataset_name

if dataset_name == "enron":
    directory_supervised = "Enron/"

if osName == "WINDOWS":
    os.environ['SPARK_HOME'] = "C:/Mine/Spark/spark-1.4.1-bin-hadoop2.6"
    sys.path.append("C:/Mine/Spark/spark-1.4.1-bin-hadoop2.6/python")
    sys.path.append('C:/Mine/Spark/spark-1.4.1-bin-hadoop2.6/python/pyspark')
    os.environ['HADOOP_HOME'] = "C:/Mine/Spark/hadoop-2.6.0"
    sys.path.append("C:/Mine/Spark/hadoop-2.6.0/bin")

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext
from pyspark.sql.types import *
sc = SparkContext()
sqlContext = SQLContext(sc)
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import BinaryClassificationMetrics,MulticlassMetrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pylab import title,gcf

def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

# This function determines which classification method must be used
# based on user input (SVM or RandomForest)
def classifyData(trainSetFile,testSetFile,classifier,activeFeatures):

    print("STEP : In Classification")

    if classifier == "SVM":
       return svmClassification(trainSetFile,testSetFile)

    elif classifier == "RF":
        randomForestClassification(trainSetFile,testSetFile,activeFeatures)

# This function does SVM classification on data
def svmClassification(trainSetFile,testSetFile):

    data1 = sc.textFile(directory_supervised + trainSetFile)
    trainData = data1.map(parsePoint)
    data2 = sc.textFile(directory_supervised + testSetFile)
    testData = data2.map(parsePoint)

    # Build the model
    model = SVMWithSGD.train(trainData, iterations=10)

    # Evaluating the model on training data
    '''labelsAndPreds = trainData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(trainData.count())
    print("Training Error = " + str(trainErr))
    labelsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
    testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testData.count())
    print("Test Error = " + str(testErr))
    return testErr'''
    #labelsAndPreds = testData.map(lambda p: (p.label, float(model.predict(p.features))))
    #truePos = labelsAndPreds.filter(lambda p: p[0] == p[1]).count()
    #print("True pos : " + str(truePos))
    #metrics1 = MulticlassMetrics(labelsAndPreds)
    #print("Recall : " + str(metrics1.recall()))
    #print("Precision : " + str(metrics1.precision()))
    #print(metrics1.confusionMatrix())

    model.clearThreshold()
    scoreAndLabels = testData.map(lambda p: (float(model.predict(p.features)), p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    return metrics.areaUnderROC

# This function calculates feature importance using RandomForest classifier
def randomForestClassification(trainSetFile,testSetFile,activeFeatures):

    activeFeatureNames = ["AA","JC","CN","PA","SP","ID1","ID2","OD1","OD2","PR1","PR2","BC","DR"]
    newColumns = ["Class"]
    i = 0
    for f in activeFeatures:
        if f == 1:
            newColumns.append(activeFeatureNames[i])
        i += 1

    trainData = pd.read_csv(directory_supervised + trainSetFile,sep = " ")
    trainData.columns = newColumns


    forest = RandomForestClassifier(n_jobs=6,oob_score=True,max_features=activeFeatures.count(1)-2,n_estimators=10)
    newColumns.remove("Class")
    newColumns.remove("SP")
    newColumns.remove("BC")
    X = trainData[newColumns]
    #X.drop('SP', axis=1, inplace=True)
    y = trainData.Class

    forest.fit(X, y)
    print "OOB Score:",forest.oob_score_

    fi = pd.DataFrame({'feature':X.columns.tolist(),'imp':forest.feature_importances_})
    fi.index = X.columns.tolist()

    fi.sort('imp',ascending=True).imp.plot(kind='barh')
    title('Feature Importance\n',fontsize=24)
    f = gcf()
    f.set_size_inches(7, 10)
    plt.xlabel('Importance', fontsize=18)
    plt.ylabel('Feature Name', fontsize=16)
    plt.show()
