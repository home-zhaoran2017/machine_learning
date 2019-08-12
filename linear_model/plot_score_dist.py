import numpy as np
import matplotlib.pyplot as plt
from py_tools import gen_kde

data = np.loadtxt("score.txt")
x0,y0=gen_kde(data[:,0],1000,0.2,1.0)
x1,y1=gen_kde(data[:,1],1000,0.2,1.0)

plt.figure(dpi=100)
plt.subplots_adjust(hspace=0.3)

plt.subplot(2,1,1)
plt.plot(x0,y0)
plt.title("score on train set")

plt.subplot(2,1,2)
plt.plot(x1,y1,color="darkred")
#plt.title("score on test set")

plt.savefig("pic.png")
