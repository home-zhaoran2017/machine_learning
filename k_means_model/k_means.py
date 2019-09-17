import time
import numpy as np
import pandas as pd

def distEclud(vecA, vecB):
    return np.sqrt(np.sum((vecA-vecB)**2))

def randCenter(data, k):
    nFeat = data.shape[1]
    centroids = np.zeros((k,nFeat))
    for j in range(nFeat):
        minJ = np.min(data[:,j])
        maxJ = np.max(data[:,j])
        rangeJ = maxJ - minJ
        centroids[:,j] = minJ + rangeJ * np.random.rand(k)
    return centroids

def kMeans(data, k):
    N = data.shape[0]
    clusterAssment = np.zeros((N,2))
    centroids = randCenter(data, k)
    clusterChanged = True
    step = 0
    start_time = time.time()
    while clusterChanged:
        step += 1

        clusterChanged = False
        for i in range(N):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distEclud(centroids[j,:], data[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2

        for cent in range(k):
            emptyId=[]
            ptsInClust = data[clusterAssment[:,0]==cent]
            if ptsInClust.tolist():
                centroids[cent,:] = np.mean(ptsInClust, axis=0)
            else:
                emptyId.append(cent)

        # 处理空簇，随机选择一个簇，随机选择该簇中距离中心最远的前五个样本中的一个加入空簇中
        Id_tmp = []
        for cent in emptyId:
            while True:
                choiceClust = np.random.choice(np.unique(clusterAssment[:,0]))
                maxD_list = np.sort(clusterAssment[clusterAssment[:,0]==choiceClust][:,1]).tolist()[-5:]
                np.random.shuffle(maxD_list)
                maxD = maxD_list[0]
                Id = np.where((clusterAssment[:,0]==choiceClust) & (clusterAssment[:,1]==maxD))[0][0]
                if Id not in Id_tmp:
                    clusterAssment[Id,:] = cent, 0
                    centroids[cent,:] = data[Id]
                    Id_tmp.append(Id)
                    break

        end_time = time.time()
        print("\rStep: %d; Using Time: %.2f s"%(step,end_time-start_time),end='')


    clusterAssment = pd.DataFrame(clusterAssment,columns=["clusID","squareDis"])
    clusterAssment["clusID"]=clusterAssment["clusID"].astype(int)
    return centroids, clusterAssment
