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

#output=open('data_{0}.dat'.format(comm_rank),'w')
 #divide the data to each processor
local_data_offset = np.linspace(0, row, comm_size + 1).astype('int')
   
data1 = zeros([row,column])
for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        data1[i,j] = origin[i,j]
np.savetxt('data_{}.dat'.format(comm_rank),data1)


local_origin = np.loadtxt('data_{}.dat'.format(comm_rank))


#SVD
U, s, Vh = linalg.svd(origin)#, full_matrices = False)
U1, s1, Vh1 = linalg.svd(local_origin)#, full_matrices = False)
#print s1

s_fix = np.zeros(len(s1))
mean = np.mean(s)
median = np.median(s)
for i in range(len(s1)):
    if s1[i] > median:
        s_fix[i] = s1[i]
#print s_fix
#print median
S1 = linalg.diagsvd(s_fix, U1.shape[1], Vh1.shape[0])
SVD = np.dot(U1, np.dot(S1, Vh1))
np.savetxt("SVD_{}.dat".format(comm_rank),SVD)
#print U1.shape[1], Vh1.shape[0]

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