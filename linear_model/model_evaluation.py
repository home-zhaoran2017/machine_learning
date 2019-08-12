import numpy as np

def adj_r_squared(Y,Y_hat,p):
    rsq = r_squared(Y,Y_hat)
    n = len(Y)
    adRsq = 1 - (1-rsq)*(n-1)/(n-p-1)

    return adRsq

def r_squared(Y, Y_hat):
    A=np.sum((Y-Y_hat)**2)
    meanY = np.mean(Y)
    B=np.sum((Y-meanY)**2)

    return 1-A/B

def medianAE(Y, Y_hat):
    return np.median(np.abs(Y-Y_hat))

def msle(Y, Y_hat):
    return np.mean(np.log(1+Y)-np.log(1+Y_hat))**2

def mae(Y, Y_hat):
    return np.mean(np.abs(Y_hat - Y))

def mse(Y, Y_hat):
    return np.mean((Y_hat - Y)**2)

def rmse(Y, Y_hat):
    return np.sqrt(mse(Y, Y_hat))

def ev(Y, Y_hat):
    A = np.std(Y-Y_hat)**2
    B = np.std(Y)**2
    return 1-A/B
