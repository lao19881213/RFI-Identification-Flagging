import numpy as np
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI
from numpy import *
from scipy import linalg
#This method reduce the noise directly, thus RFI cannot be flagged or outputed. Only the whole fixed data can be saved.

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#import data
def loadtxt_SVD(data_file):
    origin = np.loadtxt(data_file)
    return origin
    
origin = loadtxt_SVD('samplerfi.dat')
row = origin.shape[0]
column = origin.shape[1]

#SVD
U, s, Vh = linalg.svd(origin)#, full_matrices = False)

 #divide the data to each processor
local_data_offset = np.linspace(0, row, comm_size + 1).astype('int')
   
data1 = zeros(len(s)).reshape(-1,1)
for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    data1[i] = s[i]
np.savetxt('data_{}.dat'.format(comm_rank),data1)

local_origin = np.loadtxt('data_{}.dat'.format(comm_rank))



s_fix = np.zeros(len(s))
mean = np.mean(s)
median = np.median(s)
for i in arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    if local_origin[i] > median:
        s_fix[i] = local_origin[i]

S = linalg.diagsvd(s_fix, U.shape[1], Vh.shape[0])

SVD = np.dot(U, np.dot(S, Vh))
np.savetxt("SVD_{}.dat".format(comm_rank),SVD)
#print U.shape[1], Vh.shape[0]

#plot
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def plt_1(m,n):
    plt.colorbar(n)
    plt.xlabel('origin')
    forceAspect(m,aspect=1)
    
def plt_2(m,n):
    plt.colorbar(n)
    plt.xlabel('SVD')
    forceAspect(m,aspect=1)
    
fig1 = plt.figure()
ax = fig1.add_subplot(121)
im=ax.imshow(origin)
plt_1(ax,im)
  
ax = fig1.add_subplot(122)
im=ax.imshow(SVD)
plt_2(ax,im)

plt.show()