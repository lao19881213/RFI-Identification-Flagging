import numpy as np
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI
import time
from numpy import *
from scipy import linalg
from RFI_flag import *

print time.localtime()

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#import data
origin = np.loadtxt('samplerfi.dat')
row = origin.shape[0]
column = origin.shape[1]

#output=open('data_{0}.dat'.format(comm_rank),'w')
 #divide the data to each processor
local_data_offset = np.linspace(0, column, comm_size + 1).astype('int')
   
data1 = zeros([row,column])
for i in np.arange(row):
    for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
        data1[i,j] = origin[i,j]
np.savetxt('data_{}.dat'.format(comm_rank),data1)


local_origin = np.loadtxt('data_{}.dat'.format(comm_rank))

x = RFI_flag()
x.singular_value_decomposition(origin,local_origin,comm_rank)

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
im=ax.imshow(RFI_flag.SVD)
plt.colorbar(im)
plt.xlabel('SVD')
forceAspect(ax,aspect=1)
plt.savefig('RFI_%s.png' %str(comm_rank)) 
#plt.show()
print time.localtime()