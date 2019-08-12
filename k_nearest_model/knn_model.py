import numpy as np

class KnnModel():
    def __init__(self, k, alpha):
        self.k = k
        self.alpha = alpha

    def load_dataSet(self,X,Y):
        self.data = X
        self.target=Y

    def _distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def _find_nbor(self,x):
        listD = np.zeros((self.data.shape[0],2))
        for i,d in enumerate(self.data):
            listD[i,0]=i
            listD[i,1]=self._distance(x,d)

        index=np.argsort(listD[:,1])[:self.k]
        ind=np.array(listD[index,0],dtype=int)
        dis=listD[index,1]
        dis=np.array([dis]).T

        return ind, dis

    def _decision(self,ind,dis):
        v = self.target[ind]
        w = np.exp(-self.alpha*dis)
        w = w/np.sum(w)
        return np.sum(w*v)

    def _predict_one(self,x):
        nborId, nborDis = self._find_nbor(x)
        y = self._decision(nborId,nborDis)
        return y 

    def predict(self,X):
        Y=[[self._predict_one(x)] for x in X]
        return np.array(Y)
