from sklearn.cluster import KMeans
from sklearn import metrics

class KMeansModel():
    def __init__(self,nClusters=3):
        self.kmeans_model=KMeans(n_clusters = nClusters)

    def fit(self,X):
        self.kmeans_model.fit(X)
        
    def predict(self,X):
        y = self.kmeans_model.predict(X)
        
        return y
    
    def score(self, X, y):
        score = metrics.calinski_harabasz_score(X, y)
        return score
