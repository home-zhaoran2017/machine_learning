import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fpic = sys.argv[0][:-3]
data = pd.read_csv("../boston_house_price.txt",sep='|')

y,x = np.histogram(data["TAX"],range=[100,800],bins=7,density=True)
x=x[1:]

plt.figure(dpi=100,figsize=(6.4,3.6))
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.bar(x-50,y,width=50)
plt.xticks(ticks=[100]+list(x))
plt.xlabel("TAX")
plt.ylabel("P")
plt.savefig(fpic+'.png')
