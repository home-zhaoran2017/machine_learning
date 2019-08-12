import sys
import numpy as np
import pandas as pd
from knn_model import KnnModel
from model_evaluation import r_squared

data = pd.read_csv("boston_house_price.txt",sep='|')
data = data[["LSTAT","RM","PTRATIO","INDUS","MEDV"]].values
data = (data-np.min(data))/(np.max(data)-np.min(data))
N = data.shape[0]
k=3
sigma=0.05
alpha=1/sigma

for step in range(100):
    np.random.shuffle(data)
    train = data[:int(N*0.7),:]
    test = data[int(N*0.7):,:]

    X_train = train[:,:-1]
    Y_train = train[:,[-1]]

    X_test = test[:,:-1]
    Y_test = test[:,[-1]]

    model = KnnModel(k,alpha)
    model.load_dataSet(X_train,Y_train)

    hatY_train = model.predict(X_train)
    hatY_test = model.predict(X_test)

    r1 = r_squared(Y_train,hatY_train)
    r2 = r_squared(Y_test,hatY_test)

    print("%2d %5.2f %.9f %.9f"%(k,sigma,r1,r2))
    sys.stdout.flush()
