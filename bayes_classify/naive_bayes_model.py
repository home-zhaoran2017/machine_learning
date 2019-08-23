import sys
import numpy as np

def model_training(dataSet):
    X = dataSet[:,:-1]
    Y = dataSet[:,-1]
    nCluster = len(np.unique(Y))
    nFeature = X.shape[1]

    bayes_ptable=np.zeros((nFeature+1,nCluster))

    for cluster in np.unique(Y):
        xCluster=X[dataSet[:,-1]==cluster]
        nSamplesC=xCluster.shape[0]
        bayes_ptable[0,cluster]=nSamplesC/X.shape[0]
        nSamplesC+=2
        for k in range(xCluster.shape[1]):
            count=np.sum(xCluster[:,k]==0)+1
            bayes_ptable[k+1,cluster]=count/nSamplesC

    np.savetxt("bayes_table.txt",bayes_ptable)
    return

def model_predict(model,dataSet):
    prC = np.log(model[0,:])
    Table=[None]*len(prC)
    Table[0]=np.array([list(model[1:,0])]*2).T
    Table[1]=np.array([list(model[1:,1])]*2).T
    Table[0][:,1]=1.0-Table[0][:,1]
    Table[1][:,1]=1.0-Table[1][:,1]
    Table=[np.log(table) for table in Table]

    predY=[None]*len(dataSet)
    for n,X in enumerate(dataSet):
        C, maxpC = 0, -np.inf
        for c, table in enumerate(Table):
            # print(table.shape,X.shape)
            pC=np.sum(table[X==0,0])+np.sum(table[X==1,1])+prC[c]
            if pC > maxpC:
                maxpC = pC
                C=c
        predY[n]=C

    return np.array(predY)

def main():
    dataSet = np.loadtxt("corpusVec.txt",dtype=int)
    model_training(dataSet)
    model = np.loadtxt("bayes_table.txt")
    predY=model_predict(model,dataSet[:,:-1])
