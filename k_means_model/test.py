import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from k_means_model_sk import KMeansModel

data = pd.read_csv("data.txt",sep='|',header=None)
data = data.iloc[:,:-1].values

model = KMeansModel(nClusters=5)
model.fit(data)
y=model.predict(data)
score = model.score(data,y)
print(score)


x1 = data[:,0]
x2 = data[:,1]

plt.figure(dpi=100)

plt.scatter(x1,x2,s=0.5,c=y,cmap="gnuplot")
plt.colorbar()

plt.savefig("pic.png")
