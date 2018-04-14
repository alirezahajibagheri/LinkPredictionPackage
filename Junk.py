import datetime
dataFile = open("DataAnalysis/mention.txt","r")


baseDate = ""
outFile = ""
i = 1
for line in dataFile:

    elems = line.strip("\n").split(" ")
    myDate = datetime.datetime.fromtimestamp(int(elems[2])).strftime('%Y-%m-%d %H:%M:%S')
    if myDate.split(" ")[0] != baseDate:
        outFile = open("DataAnalysis/Cannes/mention-" + str(i) + ".txt" , "w")
        baseDate = myDate.split(" ")[0]
        i += 1

    outFile.write(elems[0] + " " + elems[1] + "\n")



'''
npoints = 1000
x, y = np.random.normal(10, 2, (2, npoints))

fig, ax = plt.subplots()
artist = ax.hexbin(x, y, gridsize=20, cmap='gray_r', edgecolor='white')

# Create the inset axes and use it for the colorbar.
cax = fig.add_axes([0.8, 0.15, 0.05, 0.3])
cbar = fig.colorbar(artist, cax=cax)

plt.show()

conn = pymysql.connect(host=dbIP,
                port=dbPort,user=dbUsername, passwd=dbPassword,
                db="enron")

idsQuery = "SELECT userid,email FROM members"
ids = {}
cursor = conn.cursor()
cursor.execute(idsQuery)
result = cursor.fetchall()
for row in result:
    ids[row[1]] = row[0]

i = 1
for year in range(2000,2002):
    for month in range(1,13):
        print(i)
        fileName = "enron-" + str(i)
        file = open("Enron/"+fileName,'w')

        query = "SELECT DISTINCT sender, receiver FROM filteredmessage WHERE YEAR(FROM_UNIXTIME(unixdate))="+ str(year) +" AND MONTH(FROM_UNIXTIME(unixdate))=" + str(month)
        cursor.execute(query)
        result = cursor.fetchall()
        for row in result:
            file.write(str(ids.get(row[0])) + " " + str(ids.get(row[1])))
            file.write("\n")

        fileName = "enron-full-" + str(i)
        file = open("Enron/"+fileName,'w')
        query = "SELECT DISTINCT sender, receiver FROM filteredmessage WHERE FROM_UNIXTIME(unixdate, '%Y-%m-%d' )>='2000-01-01' AND FROM_UNIXTIME(unixdate, '%Y-%m-%d' )<='"+str(year)+ "-" + str(month) + "-31'"
        print(query)
        cursor.execute(query)
        result = cursor.fetchall()
        for row in result:
            file.write(str(ids.get(row[0])) + " " + str(ids.get(row[1])))
            file.write("\n")

        i+=1

# Function implemented to create train sets
import sys
import csv
import igraph
from igraph import *
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt, savetxt
from sklearn import svm
from sklearn import datasets
import arff, numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
#import pylab as pl

def classifyData(trainSetFile,testSetFile,classifier,numOfFeatures):
    print("STEP: Classification")

    trainSet = pd.read_csv("Networks/"+trainSetFile)
    print(trainSet.head(50))

    trainSet = arff.load(open("Networks/"+trainSetFile, 'r'))
    trainData = np.array(trainSet['data'])

    testSet = arff.load(open("Networks/"+testSetFile, 'r'))
    testData = np.array(testSet['data'])

    X_train = trainData[:,0:numOfFeatures].astype(np.float)
    X_train = X_train / X_train.max(axis=0)
    y_train = trainData[:,numOfFeatures]
    X_test = testData[:,0:numOfFeatures].astype(np.float)
    X_test = X_test / X_test.max(axis=0)
    y_test = testData[:,numOfFeatures]
    y_test = y_test.astype(np.int)

    auroc = 0
    if classifier=="SVM":
        auroc = svmClassifier(X_train,y_train,X_test,y_test)

    return auroc

def svmClassifier(X_train,y_train,X_test,y_test):

    print("STEP : SVM set as classifier")
    #print(X_train)
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,shrinking=True, tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    print("STEP : Classifier fit to train data")
    y_predict = clf.predict(X_test)
    y_predict = y_predict.astype(np.int)
    print("STEP : Prediction is done !")

    # In case percision, recall or fscore is needed, uncomment
    #percision,recall,fscore,support = precision_recall_fscore_support(y_test,y_predict,average='weighted')
    #print(percision)
    #print(recall)
    #print(fscore)

    # Calculating fpr and tpr to output AUROC
    probas_ = clf.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %f" % roc_auc)
    return roc_auc

from igraph import *
import Adamic_Adar
import Common_Neighbors
import Jaccard_Coefficient
import numpy as np
import Create_Networks_CSV
from sklearn import preprocessing
import pandas as pd
min_max_scaler = preprocessing.MinMaxScaler()

tradeTrainDf = pd.read_csv("Networks/11.txt")
x_scaled = min_max_scaler.fit_transform(tradeTrainDf.values)
tradeTrainDf = pd.DataFrame(x_scaled)
tradeTrainDf.to_csv("Networks/nima2.txt", sep=" ", index=False, header=False)

def read_csv(fileName,sep,tableName):
    file = sc.textFile(fileName)
    all = file.map(lambda line: line.split(sep))
    first_line = file.first()

    header = first_line.split(sep)
    no_header = all.filter(lambda line: line[0] != header[0])

    fields = [StructField(field_name, StringType(), True) for field_name in header]
    schema = StructType(fields)

    df = sqlContext.createDataFrame(no_header, schema)

    df.registerTempTable(tableName)

    return df
    trainData = pd.read_csv("Networks/" + trainSetFile)
    testData = pd.read_csv("Networks/" + testSetFile)

    #forest = RandomForestClassifier(n_jobs=6,oob_score=True,max_features=4,n_estimators=10)
    forest = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,shrinking=True, tol=0.001, verbose=False)
    X_train = trainData[['AA','JC','CN','PA']]
    y_train = trainData.Class

    print("STEP : Classifier fit to train data")
    forest.fit(X_train, y_train)
    #print "OOB Score:",forest.oob_score_


    X_test = testData[['AA','JC','CN','PA']]
    y_test = testData.Class
    y_predict = forest.predict(X_test)
    y_predict = y_predict.astype(np.int)
    print("STEP : Prediction is done !")
    # Calculating fpr and tpr to output AUROC
    probas_ = forest.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %f" % roc_auc)
    return roc_auc
'''
'''
columns = ['A','B','C']

xx = np.zeros(shape=(1,3))
df = pd.DataFrame(xx,columns=columns)
print(df.head())

g = Graph()

g.add_vertices(3)
g.to_directed()
g.add_edges([(0,1),(0,2),(2,0)])
print(g)
Matrix = [[0 for x in range(3)] for x in range(3)]
print(Matrix)
for edge in g.es:
    Matrix[edge.tuple[0]][edge.tuple[1]]=1

print(Matrix)

adamic = Adamic_Adar.adamic_adar_score(g)
adamic = np.array(adamic)

print(adamic)


nn = open("Networks/trades-train-network.txt",'r')
graph = Graph.Read_Ncol(nn, names=True, weights="if_present", directed=True)
aa = graph.get_adjacency()
print(aa.shape[0])
JC = Jaccard_Coefficient.jaccard_coefficient_score(graph)

f = open("nima.txt","w")
tot = 0
for i in range(0,aa.shape[0]):
    for j in range(0,aa.shape[1]):
        if JC[i][j]>=1.0 and aa[i][j]==1:
            tot +=1
            #f.write(str(JC[i][j]) + "\n")

print(tot)
'''

