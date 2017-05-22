#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI
import time
#import matplotlib.image as mpimg
from astropy.convolution import Gaussian2DKernel, convolve_fft
from PIL import Image
from pylab import *
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from RFI_flag import *

print time.localtime()

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

    
origin = np.loadtxt('samplerfi.dat')
# 2D Gaussian Filtering
#RFI = array(Image.open('rfidataSnapshot.png').convert('L'))  #this line read the image data
f = Gaussian2DKernel(stddev=0.5)
filtered = convolve_fft(origin, f) #boundary='extend')
row = filtered.shape[0]
column = filtered.shape[1]
RFI = ['Row', 'Colunm','Power' ]
np.savetxt('2DGaussianFiltering.dat',filtered)

#divide the data to each processor
local_data_offset = np.linspace(0, column, comm_size + 1).astype('int')
   
#get the local data which will be processed in this processor
local_data = filtered[local_data_offset[comm_rank] :local_data_offset[comm_rank + 1]]

data1 = zeros([row,column])
for i in np.arange(row):
    for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
        data1[i,j] = filtered[i,j]
np.savetxt('2DGaussianFiltering_{}.dat'.format(comm_rank),data1)

local_filtered = np.loadtxt('2DGaussianFiltering_{}.dat'.format(comm_rank))
flagged_gf = convolve_fft(local_filtered, f)

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(filtered)-np.min(filtered))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(filtered), np.max(filtered), step)

x = RFI_flag()
x.gaus_filter(filtered,local_filtered,local_data_offset,comm_rank,row,column,RFI,bins,x_hist,y_hist,step,flagged_gf)


#plot the histogram fitting & data RFI
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

fig1 = plt.figure()
bx=plt.gca()
bx.set_xlim(0,10)
bx.set_ylim(0,100)
plt.plot(x_hist, y_hist, 'k.', label = 'filtered')
plt.plot(x_hist, RFI_flag.h(x_hist), label = 'fitting',color = 'red')
plt.xlabel('Power')
plt.ylabel('Number')
plt.legend()
plt.savefig('rfi_%s.png' %str(comm_rank))  

fig2 = plt.figure()
ax = fig2.add_subplot(131)
#cmap = mpl.cm.hot
im=ax.imshow(origin)#, cmap=cmap)
plt.colorbar(im)
plt.xlabel('origin')
#plt.axis('off')
forceAspect(ax,aspect=1)

ax = fig2.add_subplot(132)
#cmap = mpl.cm.hot
im=ax.imshow(filtered)#, cmap=cmap)
plt.colorbar(im)
plt.xlabel('filtered')
#plt.axis('off')
forceAspect(ax,aspect=1)

ax = fig2.add_subplot(133)
#cmap = mpl.cm.hot
im=ax.imshow(flagged_gf)#, cmap=cmap)
plt.colorbar(im)
plt.xlabel('RFI flagged')
#plt.axis('off')
forceAspect(ax,aspect=1)
plt.savefig('RFI_%s.png' %str(comm_rank))  
#plt.show()
print time.localtime()