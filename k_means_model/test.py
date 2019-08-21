import numpy as np
import pandas as pd
from k_means_model_sk import KMeansModel

data = pd.read_csv("data.txt",sep='|',header=None)
data = data.iloc[:,:-1].values

model = KMeansModel(nClusters=5)
model.fit(data)
y=model.predict(data)
score = model.score(data,y)
print(score)
