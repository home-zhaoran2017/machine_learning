import time
import numpy as np
from numpy import random
from numpy import linalg
import pandas as pd


class GMM():
    def __init__(self, K, nFeat):
        self.K=K
        self.nFeat=nFeat
        self.ALPHA=random.random(K).tolist()
        self.MU = random.random((K,nFeat)).tolist()
        
        self.SIGMA=[]
        for k in range(K):
            self.SIGMA.append(np.eye(nFeat)*random.random(nFeat).tolist())
            
        self.GAMMA=None

    def _multiNorm(self, x, mu, sigma):
        x = np.mat(x).T
        mu = np.mat(mu).T
        sigma = np.mat(sigma)
        p = self.nFeat
        sigmaAbs = linalg.det(sigma)

        return 1/((2.0*np.pi)**(p/2)*sigmaAbs**1/2)*np.exp(-0.5*(x-mu).T*sigma.I*(x-mu))
    
    def _max_variation(self, ALPHA_TMP, MU_TMP, SIGMA_TMP):
        max_alpha = np.max(np.array(self.ALPHA)-np.array(ALPHA_TMP))
        max_mu = np.max(np.array(self.MU)-np.array(MU_TMP))
        max_sigma = np.max(np.array(self.SIGMA)-np.array(SIGMA_TMP))
        
        return np.max([max_alpha, max_mu, max_sigma])
    
    def expectation(self, data):
        GAMMA=np.zeros((data.shape[0],self.K))
        for j,x in enumerate(data):
            for k in range(self.K):
                alpha=self.ALPHA[k]
                sigma=self.SIGMA[k]
               
                # 处理协方差矩阵奇异的情况，此时随机选择一个样本点的特征值作为该高斯分布的均值，并随机生成一个新的
                # 协方差矩阵，而 gamma 值不变。
                if linalg.det(np.mat(sigma)) == 0:
                    ide = np.random.randint(data.shape[0])
                    self.MU[k]= data[ide].tolist()
                    self.SIGMA[k]=np.eye(self.nFeat)*random.random(self.nFeat).tolist()
                else:
                    mu=self.MU[k]
                    GAMMA[j,k]=alpha*self._multiNorm(x,mu,sigma)
                
        gamaSUM = np.sum(GAMMA,axis=1)
        gamaSUM = gamaSUM.reshape((gamaSUM.size,1))
        self.GAMMA = GAMMA/gamaSUM

    def maximization(self, data):
        p = self.nFeat

        self.ALPHA = np.mean(self.GAMMA,axis=0).tolist()

        for k in range(self.K):
            mu = np.mean(self.GAMMA[:,[k]]*data,axis=0)/self.ALPHA[k]
            self.MU[k]=mu.tolist()

        for k in range(self.K):
            sigma=np.zeros((p,p))
            for j,x in enumerate(data):
                gamma = self.GAMMA[j,k]
                mu = np.mat(self.MU[k]).T
                d = np.mat(x).T - mu
                sigma = sigma + np.array(gamma*d*d.T)

            sigma = sigma / data.shape[0]
            sigma = sigma / self.ALPHA[k]
            self.SIGMA[k]=sigma.tolist()

    def training(self, data, eps):
        t1 = time.time()
        step = 0
        while True:
            step += 1
            ALPHA_TMP = self.ALPHA
            MU_TMP = self.MU
            SIGMA_TMP = self.SIGMA
            
            self.expectation(data)
            self.maximization(data)
            
            max_vari = self._max_variation(ALPHA_TMP, MU_TMP, SIGMA_TMP)
            t2 = time.time()
            print("\rSTEP %d; maxVari: %.9f; USED TIME: %.2f s"%(step, max_vari, t2-t1),end='')
            if max_vari < eps:
                t2 = time.time()
                print("\nEND TRAIN; USED TIME: %.2f s"%(t2-t1))
                break
                
    def clusters(self):
        assessment = pd.DataFrame(columns=["clusID","probability"])
        for prob in self.GAMMA:
            clusID = np.where(prob==np.max(prob))[0][0]
            probability = prob[clusID]
            assessment = assessment.append({"clusID":clusID,"probability":probability},ignore_index=True)
            
        assessment["clusID"] = assessment["clusID"].astype(int)
        
        return assessment
        
