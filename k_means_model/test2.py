import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmeans_model_sk import KMeansModel

data = pd.read_csv("city_info.txt",sep='\t',header=None)
data = data.iloc[:,[0,1]].values

x = data[:,0]
y = data[:,1]

model = KMeansModel(nClusters=30)
model.fit(data)
clusters = model.predict(data)
score = model.score(data,clusters)
print("Score: ",score)

plt.figure(dpi=100)
plt.plot(x,y,'.',markersize=0.5)
plt.savefig("pic2_0.png")
plt.clf()

plt.figure(dpi=100)
plt.scatter(x,y,s=0.5,c=clusters,cmap="gnuplot")
plt.colorbar()
plt.savefig("pic2_1.png")
plt.clf()
