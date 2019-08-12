import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fpic = sys.argv[0][:-3]
data = pd.read_csv("../boston_house_price.txt",sep='|')

y,x = np.histogram(data["RAD"],range=[0,30],bins=30,density=True)
x=x[1:]

plt.figure(dpi=100,figsize=(6.4,3.6))
plt.subplots_adjust(bottom=0.15,left=0.1)
plt.bar(x-0.5,y,width=0.5)
plt.xticks(ticks=[0]+list(x))
plt.xlabel("RAD")
plt.ylabel("P")
plt.savefig(fpic+'.png')
