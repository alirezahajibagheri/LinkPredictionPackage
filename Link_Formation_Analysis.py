from __future__ import division
from igraph import *
import pymysql
import pymysql.cursors
# This could be replaced with your own database information or you
# can simply read info of graphs from a file
from Configurations import dbIP,dbPort,dbName,dbUsername,dbPassword,directory_supervised,dataset_name
from numpy import zeros
from datetime import timedelta
from datetime import datetime

conn = pymysql.connect(host=dbIP,
                port=dbPort,user=dbUsername, passwd=dbPassword,
                db=dbName)


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

baseDateTrain = "2009-12-01"
baseDateTest = "2009-12-02"
baseDateTrain_Enron = "1"
baseDateTest_Enron = "2"
if dataset_name == "enron":
    directory = "Enron/"
else:
    directory = directory_supervised

def useDailyRate(graph):

    dailyRate = []

    return dailyRate

def averageTimeSeriesRate(graph,date,name,mode):

    dailyRate = zeros(graph.vcount())
    dailyForecast = zeros(graph.vcount())
    if mode == "train":
        baseDate = baseDateTrain
    if mode == "test":
        baseDate = baseDateTest
    diff = days_between(baseDate,date)
    for v in range(graph.vcount()):
        n = 0
        while n <= diff:
            new_date = datetime.strptime(baseDate, "%Y-%m-%d") + timedelta(days=n)
            file = name + "-"+ mode +"-network-" + str(new_date.date()) + ".txt"
            netFile = open(directory+file,'r')
            new_graph = Graph.Read_Ncol(netFile, names=True, weights="if_present", directed=True)
            try:
                index = new_graph.vs.find(name=graph.vs[v]["name"]).index
                dailyRate[v] = dailyRate[v] + new_graph.degree(index,mode='OUT')
            except ValueError:
                dailyRate[v] = dailyRate[v] + 0
            n += 1
        dailyForecast[v] = dailyRate[v] / n
    return dailyForecast

def averageTimeSeriesRate_Enron(graph,date,name,mode):

    dailyRate = zeros(graph.vcount())
    dailyForecast = zeros(graph.vcount())
    if mode == "train":
        baseDate = baseDateTrain_Enron
    if mode == "test":
        baseDate = baseDateTest_Enron
    diff = int(date) - int(baseDate)
    for v in range(graph.vcount()):
        n = 0
        while n <= diff:
            new_date = int(baseDate) + n
            file = name + "-"+ str(new_date)
            netFile = open(directory+file,'r')
            new_graph = Graph.Read_Ncol(netFile, names=True, weights="if_present", directed=True)
            try:
                index = new_graph.vs.find(name=graph.vs[v]["name"]).index
                dailyRate[v] = dailyRate[v] + new_graph.degree(index,mode='OUT')
            except ValueError:
                dailyRate[v] = dailyRate[v] + 0
            n += 1
        dailyForecast[v] = dailyRate[v] / n
    return dailyForecast

def movingAverageTimeSeriesRate(graph,date,name,mode):

    move = 3
    dailyRate = zeros(graph.vcount())
    dailyForecast = zeros(graph.vcount())
    if mode == "train":
        baseDate = baseDateTrain
    if mode == "test":
        baseDate = baseDateTest
    diff = days_between(baseDate,date)
    if diff > move:
        baseDate = datetime.strptime(baseDate, "%Y-%m-%d") + timedelta(days=(diff-move-1))
        baseDate = str(baseDate.date())
    for v in range(graph.vcount()):
        n = 0
        new_date = datetime.strptime(baseDate, "%Y-%m-%d") + timedelta(days=n)
        while new_date.date() <= datetime.strptime(date, "%Y-%m-%d").date():
            file = name + "-"+ mode +"-network-" + str(new_date.date()) + ".txt"
            netFile = open(directory+file,'r')
            new_graph = Graph.Read_Ncol(netFile, names=True, weights="if_present", directed=True)
            try:
                index = new_graph.vs.find(name=graph.vs[v]["name"]).index
                dailyRate[v] = dailyRate[v] + new_graph.degree(index,mode='OUT')
            except ValueError:
                dailyRate[v] = dailyRate[v] + 0
            n += 1
            new_date = datetime.strptime(baseDate, "%Y-%m-%d") + timedelta(days=n)
        dailyForecast[v] = dailyRate[v] / n
    return dailyForecast

def movingAverageTimeSeriesRate_Enron(graph,date,name,mode):

    move = 3
    dailyRate = zeros(graph.vcount())
    dailyForecast = zeros(graph.vcount())
    if mode == "train":
        baseDate = baseDateTrain_Enron
    if mode == "test":
        baseDate = baseDateTest_Enron
    diff = int(date) - int(baseDate)
    if diff > move:
        baseDate = int(baseDate) + (diff-move-1)
        baseDate = str(baseDate)
    for v in range(graph.vcount()):
        n = 0
        new_date = int(baseDate) + n
        while new_date <= int(date):
            file = name + "-" + str(new_date)
            netFile = open(directory+file,'r')
            new_graph = Graph.Read_Ncol(netFile, names=True, weights="if_present", directed=True)
            try:
                index = new_graph.vs.find(name=graph.vs[v]["name"]).index
                dailyRate[v] = dailyRate[v] + new_graph.degree(index,mode='OUT')
            except ValueError:
                dailyRate[v] = dailyRate[v] + 0
            n += 1
            new_date = int(baseDate) + n
        dailyForecast[v] = dailyRate[v] / n
    return dailyForecast

def wMovingAverageTimeSeriesRate(graph,date,name,mode):

    move = 3
    alpha = 0.2
    beta = 0.3
    gamma = 0.5
    dailyRate = zeros(graph.vcount())
    dailyForecast = zeros(graph.vcount())
    if mode == "train":
        baseDate = baseDateTrain
    if mode == "test":
        baseDate = baseDateTest
    diff = days_between(baseDate,date)

    if diff > move:
        baseDate = datetime.strptime(baseDate, "%Y-%m-%d") + timedelta(days=(diff-move-1))
        baseDate = str(baseDate.date())
    for v in range(graph.vcount()):
        n = 0
        new_date = datetime.strptime(baseDate, "%Y-%m-%d") + timedelta(days=n)
        while new_date.date() < datetime.strptime(date, "%Y-%m-%d").date():
            file = name + "-"+ mode +"-network-" + str(new_date.date()) + ".txt"
            netFile = open(directory+file,'r')
            new_graph = Graph.Read_Ncol(netFile, names=True, weights="if_present", directed=True)
            try:
                index = new_graph.vs.find(name=graph.vs[v]["name"]).index
                if n == 0:
                    dailyRate[v] = dailyRate[v] + alpha * new_graph.degree(index,mode='OUT')
                if n == 1:
                    dailyRate[v] = dailyRate[v] + beta* new_graph.degree(index,mode='OUT')
                if n == 2:
                    dailyRate[v] = dailyRate[v] + gamma* new_graph.degree(index,mode='OUT')
            except ValueError:
                dailyRate[v] = dailyRate[v] + 0
            n += 1
            new_date = datetime.strptime(baseDate, "%Y-%m-%d") + timedelta(days=n)
        dailyForecast[v] = dailyRate[v] #/ move
    return dailyForecast

def wMovingAverageTimeSeriesRate_Enron(graph,date,name,mode):

    move = 3
    alpha = 0.2
    beta = 0.3
    gamma = 0.5
    dailyRate = zeros(graph.vcount())
    dailyForecast = zeros(graph.vcount())
    if mode == "train":
        baseDate = baseDateTrain_Enron
    if mode == "test":
        baseDate = baseDateTest_Enron
    diff = int(date) - int(baseDate)
    if diff > move:
        baseDate = int(baseDate) + (diff-move-1)
        baseDate = str(baseDate)
    for v in range(graph.vcount()):
        n = 0
        new_date = int(baseDate) + n
        while new_date <= int(date):
            file = name + "-" + str(new_date)
            netFile = open(directory+file,'r')
            new_graph = Graph.Read_Ncol(netFile, names=True, weights="if_present", directed=True)
            try:
                index = new_graph.vs.find(name=graph.vs[v]["name"]).index
                if n == 0:
                    dailyRate[v] = dailyRate[v] + alpha * new_graph.degree(index,mode='OUT')
                if n == 1:
                    dailyRate[v] = dailyRate[v] + beta* new_graph.degree(index,mode='OUT')
                if n == 2:
                    dailyRate[v] = dailyRate[v] + gamma* new_graph.degree(index,mode='OUT')
            except ValueError:
                dailyRate[v] = dailyRate[v] + 0
            n += 1
            new_date = int(baseDate) + n
        dailyForecast[v] = dailyRate[v] / n
    return dailyForecast

def smoothingTimeSeriesRate(graph,date,name):

    dailyRate = []

    return dailyRate



