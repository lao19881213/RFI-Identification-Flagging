import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
#This method reduce the noise directly, thus RFI cannot be flagged or outputed. Only the whole fixed data can be saved.

#import data
origin = np.loadtxt('samplerfi.dat')

#SVD
U, s, Vh = linalg.svd(origin)#, full_matrices = False)
s_fix = np.zeros(len(s))
mean = np.mean(s)
median = np.median(s)
for i in range(len(s)):
    if s[i] > median:
        s_fix[i] = s[i]
#print s_fix
#print median
#print s
S = linalg.diagsvd(s_fix, U.shape[1], Vh.shape[0])
SVD = np.dot(U, np.dot(S, Vh))
np.savetxt("SVD.dat",SVD)

#plot
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

fig1 = plt.figure()
ax = fig1.add_subplot(121)
im=ax.imshow(origin)
plt.colorbar(im)
plt.xlabel('origin')
forceAspect(ax,aspect=1)

ax = fig1.add_subplot(122)
im=ax.imshow(SVD)
plt.colorbar(im)
plt.xlabel('SVD')
forceAspect(ax,aspect=1)

plt.show()