# 自定义的一些功能函数
# Author: Ran Zhao
# Modified Time: 2019-06-05 15:48:57
# E-mail: zhaoran2018@sina.com

import time
from datetime import datetime
import numpy as np
import pandas as pd

def now_time():
    time.sleep(1)
    year=str(datetime.now().year)[2:]
    month="%02d" %datetime.now().month
    day="%02d" %datetime.now().day
    hour="%02d" %datetime.now().hour
    minute="%02d" %datetime.now().minute
    second="%02d" %datetime.now().second
    return year+month+day+hour+minute+second

def gen_hist(X,nbins):
    y,x = np.histogram(X,bins=nbins,range=[np.min(X),np.max(X)],density=True)
    x=x[:-1]
    return x,y

def gen_kde(data,nbins,xmin,xmax):
    from scipy import stats
    kernel=stats.gaussian_kde(data)
    x=np.linspace(xmin,xmax,nbins)
    y=kernel(x)
    return x,y

def dispKL(dis1,dis2):
    return np.sum(dis1*np.log(dis1/dis2))

def dispJS(dis1,dis2):
    dis1+=0.0000000000000001
    dis2+=0.0000000000000001
    KL1=dispKL(dis1,(dis1+dis2)/2)/2
    KL2=dispKL(dis2,(dis1+dis2)/2)/2

    return KL1+KL2

def pca(dataMat, topNfeat):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = np.dot(meanRemoved, redEigVects)
#   reconMat = dot(lowDDataMat, redEigVects.T) + array([meanVals])
    eigVals=np.sort(eigVals)
    eigVals=eigVals[::-1]
    eigVals=eigVals/np.sum(eigVals)
    return lowDDataMat, eigVals

def Chi_test(x,y):
    x=np.array(x)
    y=np.array(y)
    A=np.sum([y[x==0]==0])
    B=np.sum([y[x==0]==1])
    C=np.sum([y[x==1]==0])
    D=np.sum([y[x==1]==1])

    N1=A+C
    N2=B+D

    AA=(A+B)*N1/(N1+N2)
    BB=(A+B)*N2/(N1+N2)
    CC=(C+D)*N1/(N1+N2)
    DD=(C+D)*N2/(N1+N2)

    chi2=(A-AA)**2/AA+(B-BB)**2/BB+(C-CC)**2/CC+(D-DD)**2/DD

    return chi2

def F_test(x,y):
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    data=pd.DataFrame({'x':x,'y':y})

    lm = ols("y~x", data=data).fit()
    table = sm.stats.anova_lm(lm, typ=1)

    F=table["F"]["x"]
    p=table["PR(>F)"]["x"]

    return F,p

def accuracy(testY,predY):
    T=0
    num = len(testY)
    for y1,y2 in zip(testY,predY):
        if y1==y2:
            T+=1
    return T/num

def recall(testY,predY):
    P=0
    TP=0
    for y1,y2 in zip(testY,predY):
        if y1==1:
            P+=1
            if y1==y2: TP+=1
    return TP/P
