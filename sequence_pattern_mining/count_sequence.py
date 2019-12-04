import numpy as np
import pandas as pd

from hash_tree import HashTree


def hash_item(item):
    item=["%02d"%i for i in item]
    return ''.join(item)

def gen_subseq(seq,k):
    n=len(seq)
    subseq=[]
    hash_t=[]
    for i in range(n-k):
        s = seq[i:i+k]
        key = hash_item(s)
        if key not in hash_t:
            subseq.append(s)
            hash_t.append(key)
    return subseq

def seqSet2hashSet(X):
    hashSet=[]
    for s in X:
        hashSet.append(hash_item(s))
    return hashSet

def ContainedCount(X,Y,valNum,prime=23):
    """
    X: 序列数据集，必须是嵌套列表
    Y: 频繁序列集，必须是嵌套列表
    valNum: 频繁项长度
    prime: 一个质数，默认23，如果运行出现list越界错误，依次尝试换下一个质数29，31等
    直到不在出现错误
    """
    keys = seqSet2hashSet(Y)
    values = range(len(keys))
    idCk = dict(zip(keys,values))
    feature_matrix = np.zeros((len(X),len(Y)),dtype=int)
    
    tree = HashTree(prime=prime,valNum=valNum)
    for c in Y:
        tree.insert(c)
       
    NUM=len(X)
    for i in range(NUM):
        print("\r%.2f %%"%((i+1)/float(NUM)*100),end='')
        for c in gen_subseq(X[i],valNum):
            if tree.isExists(c):
                pos = idCk[hash_item(c)]
                feature_matrix[i,pos] += 1
   
    return feature_matrix

if __name__=="__main__":
    pass