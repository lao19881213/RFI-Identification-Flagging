import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from pylab import *
import scipy
from scipy.stats import kurtosis

a=np.loadtxt('samplerfi.dat')
b=scipy.stats.kurtosis(a,axis=0,fisher=False,bias=False,nan_policy='propagate') #0 is along freq, 1 is along time
#print b
c=np.zeros(len(b))
for i in range(len(b)):
    c[i]=i
plt.plot(c, b)
plt.show()
