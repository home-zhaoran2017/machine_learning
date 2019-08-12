import sys
import time
import numpy as np
from model_evaluation import r_squared

class LinearModel():
    def __init__(self,numInp):
        self.weights=np.random.randn(numInp)
        self.bias=np.random.randn()

    def _linear_regression(self,x):
        return np.sum(self.weights*x+self.bias)

    def _loss_function(self,X,Y):
        loss = np.array([(self._linear_regression(x)-y)**2 for x,y in zip(X,Y)])
        loss = np.mean(loss)/2
        return loss

    def predict(self,X):
        Y=[[self._linear_regression(x)] for x in X]
        return np.array(Y)
        
    def training(self,X,Y,alpha=0.001,eps=0.0001):
        n=0
        rsq_tmp=-np.inf
        loss_tmp=np.inf
        while True:
        #   for step in range(1):
            del_b = np.array([(self._linear_regression(x)-y) for x,y in zip(X,Y)])
            del_b = np.mean(del_b)
            del_w = np.array([(self._linear_regression(x)-y)*x  for x,y in zip(X,Y)])
            del_w = np.mean(del_w,axis=0)

            self.bias -= alpha*del_b
            self.weights -= alpha*del_w        

            max_del = np.max([np.abs(del_b),np.max(np.abs(del_w))])

            loss = self._loss_function(X,Y)
            if loss > loss_tmp:
                alpha=alpha*0.6
                n=0
            else:
                n+=1
            if n>=10:
                alpha+=alpha*0.01
                n=0

            loss_tmp=loss

        #   rsq = r_squared(Y,self.predict(X))

        #   print("\rlearning ratio: %.6f , max gradient: %.6f , loss: %.6f , rsq: %.6f"%(alpha,max_del,loss,rsq), end='')
        #   sys.stdout.flush()

            if max_del < eps:
                break
