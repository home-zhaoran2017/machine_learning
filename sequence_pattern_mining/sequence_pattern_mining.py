import operator
import sys

import numpy as np
import pandas as pd

from hash_tree import HashTree

def hash_item(item):
    item=["%03d"%i for i in item]
    return ''.join(item)

def save_large_items(L1, hash_tables, fname):
    with open(fname,'w') as f:
        for item in L1:
            frequence = hash_tables[hash_item(item)]
            for i in item:
                f.write("%2d,"%i)
            f.write("|%8d\n"%frequence)

def gen_candidate(Lk,k):
    Ck_new = []
    for item1 in Lk:
        for item2 in Lk:
            s1 = item1[1:]
            s2 = item2[:-1]
            if operator.eq(s1,s2):
                Ck_new.append(item1+[item2[-1]])

    keys = [hash_item(C) for C in Ck_new]
    values = [0] * len(Ck_new)
    Hk_new=dict(zip(keys,values))

    return Ck_new, Hk_new

def gen_tree(Ck,prime,valNum):
    tree = HashTree(prime=prime, valNum=valNum)
    for C in Ck:
        tree.insert(C)
    return tree

def count_frequence(data, tree, Hk):
    N = len(data)
    for n, d in enumerate(data):
        print("\r%.2f %%"%(100*(n+1)/float(N)),end='')
        res = tree.isContained(d)
        for item in res:
            Hk[hash_item(item)]+=1
    print('')
            
    return Hk

def gen_next_items(Ck,Hk,min_support):
    Lk=[]
    for C in Ck:
        if Hk[hash_item(C)] >= min_support:
            Lk.append(C)
            
    return Lk

#--------------------------------------------------------------------------
if __name__=="__main__":
    data = [line.strip().split(',') for line in open("data.txt",'r').readlines()]
    data = [[int(float(i)) for i in line] for line in data]

    k=1
    print("------------------------------")
    print("start to count %d-items set..."%k)
    C1 = [[item] for item in range(13)]
    keys = [hash_item(C) for C in C1]
    values = [0] * len(C1)
    H1=dict(zip(keys,values))
    T1 = gen_tree(C1,prime=13,valNum=1)
    H1 = count_frequence(data,T1,H1)
    L1 = gen_next_items(C1,H1,2)
    save_large_items(L1, H1, "items0001.txt")
    
    Ck_tmp = L1
    for k in range(2,11):
        print("------------------------------")
        print("start to count %d-items set..."%k)
        fname="items%04d.txt"%k
        Ck,Hk = gen_candidate(Ck_tmp, k)
        Tk = gen_tree(Ck, prime=7, valNum=k)
        Hk = count_frequence(data, Tk, Hk)
        Lk = gen_next_items(Ck, Hk, 2)
        save_large_items(Lk, Hk, fname)
        Ck_tmp = Lk
