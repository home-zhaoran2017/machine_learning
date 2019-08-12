import numpy as np
import matplotlib.pyplot as plt
from py_tools import gen_kde

data = np.loadtxt("knn_score.txt")



plt.figure(dpi=100,figsize=(3.2,6.4))

t=[5,6,7,8,9]
for i,n in enumerate(t):
    d1=data[data[:,0]==n][:,1]
    d2=data[data[:,0]==n][:,2]
    x1,y1 = gen_kde(d1,1000,0.6,1.0)
    x2,y2 = gen_kde(d2,1000,0.6,1.0)

    plt.subplot(len(t),1,i+1)

    plt.plot(x1,y1)
    plt.plot(x2,y2)


plt.savefig("pic.png")
