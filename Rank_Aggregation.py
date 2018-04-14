import numpy as np

def Borda_Rank_Aggregation(rankedLists):

    numOfRankers = len(rankedLists)
    sizeOfList = [0]*numOfRankers
    for i in range(numOfRankers):
        print(i)
        rankedLists[i] = {tuple(k): v for v, k in enumerate(rankedLists[i])}
        sizeOfList[i] = len(rankedLists[i].keys())
    members = rankedLists[0]
    scores = []
    ind = 0
    for key in members:
        score = 0
        for i in range(numOfRankers):
            score += sizeOfList[i] - rankedLists[i].get(key) - 1
        score /= numOfRankers
        scores.append(score)
        ind+=1

    # List of tuples (edges)
    members = list(members.keys())
    # Attach tuples to their score so we can sort them together
    yx = zip(scores, members)
    # Sort attached tuples,scores
    yx.sort(reverse=True)
    # List of sorted edges
    members_sorted = [x for y, x in yx]
    # Convert list of tuples to numpy array for othe functions
    members_sorted = np.asarray(members_sorted)#[list(elem) for elem in kk]
    # List of sorted scores
    scores_sorted = [y for y, x in yx]

    return members_sorted,np.array(scores_sorted)
    return 0

def Borda_Score(newList,members):
    print("STEP : Calculate Score")
    newDict = {tuple(k): v for v, k in enumerate(newList)}
    sizeOfList = len(newList)
    newScore = []
    for key in members:
        newScore.append(sizeOfList - newDict.get(key) - 1)
    return newScore

def memScoreMatch(members,scores):

    # List of tuples (edges)
    members = list(members.keys())
    # Attach tuples to their score so we can sort them together
    yx = zip(scores, members)
    # Sort attached tuples,scores
    yx.sort(reverse=True)
    # List of sorted edges
    members_sorted = [x for y, x in yx]
    # Convert list of tuples to numpy array for othe functions
    members_sorted = np.asarray(members_sorted)#[list(elem) for elem in kk]
    # List of sorted scores
    scores_sorted = [y for y, x in yx]

    return members_sorted,np.array(scores_sorted)
