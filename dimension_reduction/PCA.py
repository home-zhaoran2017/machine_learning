import numpy as np

def pca(data, ratio=0.9):
    if ratio > 1.0 or ratio < 0.0:
        print("Parameter ratio is between [0,1].")
        return

    nFeat = data.shape[1]

    meanVal = np.mean(data, axis=0)
    meanRemoved = data - meanVal
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))

    eigValsSort = np.sort(eigVals)
    eigValsSort = eigVals[::-1]
    eigValsSort = eigVals/np.sum(eigVals)

    topN=0
    eigCum=0.0
    for n in range(1,nFeat+1):
        eigCum += eigValsSort[n-1]
        if eigCum > ratio or np.abs(eigCum - ratio) < 0.00001:
            topN=n
            break

    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topN+1):-1]
    redEigVects = eigVects[:, eigValInd]

    lowDDataMat = np.dot(meanRemoved, redEigVects)
    recoDataMat = np.dot(lowDDataMat, redEigVects.T) + np.array([meanVal])

    print("principal component num: %d"%topN)

    return np.array(lowDDataMat),np.array(recoDataMat),eigValsSort[:(topN+1)]
