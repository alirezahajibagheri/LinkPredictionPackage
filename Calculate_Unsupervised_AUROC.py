from __future__ import division
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np


# Final Version
def calculate_auroc(sortedList,sortedIndices,adjacencyMatrix):

    print("STEP : Calculate AUROC")
    y_true = []
    maxVal = sortedList.max(axis=0)
    y_scores = sortedList / maxVal
    adjacencyMatrix = np.array(adjacencyMatrix.data)
    for pair in sortedIndices:
        node1 = int(pair[0])
        node2 = int(pair[1])
        y_true.append(adjacencyMatrix[node1][node2])
    auroc = roc_auc_score(y_true, y_scores)

    return auroc
