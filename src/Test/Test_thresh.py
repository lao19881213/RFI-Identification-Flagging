#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mpi4py.MPI as MPI
import time
from numpy import *
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from RFI_flag import *
#three things haven't been made sure: the value of bins, y_limit, and the fitting model(Ae^(-ax^2))

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

print time.localtime()

#import data    
origin = np.loadtxt('samplerfi.dat')
row = origin.shape[0]         #the row of origin data 
column = origin.shape[1]      #the column of origin data 
RFI = ['Row','Column','Power']

#output=open('data_{0}.dat'.format(comm_rank),'w')
 #divide the data to each processor
local_data_offset = np.linspace(0, column, comm_size + 1).astype('int')
   
data1 = zeros([row,column])
for i in np.arange(row):
    for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
        data1[i,j] = origin[i,j]
np.savetxt('data_{}.dat'.format(comm_rank),data1)


local_origin = np.loadtxt('data_{}.dat'.format(comm_rank))
flagged_th = np.loadtxt('data_{}.dat'.format(comm_rank))

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(origin)-np.min(origin))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(origin), np.max(origin), step)

x = RFI_flag()
x.thresh(origin,local_origin,local_data_offset,comm_rank,row,column,RFI,bins,step,y_hist, x_hist,flagged_th)

#plot the histogram fitting & data RFI
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

fig1 = plt.figure()
bx=plt.gca()
bx.set_xlim(0,14)
bx.set_ylim(0,100)
plt.plot(x_hist, y_hist, 'ko', label = 'origin')
plt.plot(x_hist, RFI_flag.h(x_hist), label = 'fitting',color = 'red')
plt.xlabel('Power')
plt.ylabel('Number')
plt.legend()
plt.savefig('rfi_%s.png' %str(comm_rank))

fig2 = plt.figure()
ax = fig2.add_subplot(121)
im=ax.imshow(origin)#, cmap=cmap)
plt.colorbar(im)#, cax)
plt.xlabel('origin')
forceAspect(ax,aspect=1)

ax = fig2.add_subplot(122)
im=ax.imshow(flagged_th)#, cmap=cmap)
plt.colorbar(im)
plt.xlabel('RFI')
forceAspect(ax,aspect=1)
plt.savefig('RFI_%s.png' %str(comm_rank))  
#plt.show()
print time.localtime()