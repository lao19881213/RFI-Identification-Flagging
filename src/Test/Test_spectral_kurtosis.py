import matplotlib.pyplot as plt
import numpy as np
import mpi4py.MPI as MPI
from pylab import *
import scipy
import time
from numpy import *
from scipy.stats import kurtosis
from sympy import integrate,oo,exp,symbols,pi
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from scipy.optimize import curve_fit
from scipy.special import gamma
#note: this method is not very good! It doesn't work very well for the row RFI
from RFI_flag import *

print time.localtime()

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#import data

origin = np.loadtxt('samplerfi.dat')
row = origin.shape[0]
column = origin.shape[1]

 #divide the data to each processor
local_data_offset = np.linspace(0, column, comm_size + 1).astype('int')

output = open('data{}.dat'.format(comm_rank),'w')
for i in np.arange(row):
    for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
        output.write(str(origin[i,j]) + '    ')
    output.write('\n')    
output.close()
origin1 = np.loadtxt('data{}.dat'.format(comm_rank))   
   

data1 = zeros([row,column])
for i in np.arange(row):
    for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
        data1[i,j] = origin[i,j]
np.savetxt('data_{}.dat'.format(comm_rank),data1)

local_origin = np.loadtxt('data_{}.dat'.format(comm_rank))
flagged = np.loadtxt('data_{}.dat'.format(comm_rank))

bins = 100   #if bins is smaller, than y_limit must be larger

x = RFI_flag()
x.spectral_kurtosis(origin,local_origin,local_data_offset,comm_rank,column,row,bins,flagged)

#plot
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

fig1 = plt.figure()#histogram -- Number Plot, & Pearson 4 fitting
plt.hist(RFI_flag.sk1,bins)        #sk is  axis  x     
plt.plot(RFI_flag.x_hist1, RFI_flag.h1(RFI_flag.x_hist1), label = 'fitting',color = 'red')
plt.savefig('histogram_%s.png' %str(comm_rank))  

fig2 = plt.figure()#freq -- SK Plot
c=np.zeros(len(RFI_flag.sk1))
for i in range(len(RFI_flag.sk1)):
    c[i]=i
plt.plot(c, RFI_flag.sk1, 'yo')
skflag = np.zeros(len(RFI_flag.flag_sk))
for j in np.arange(len(RFI_flag.flag_sk)):
    for i in np.arange(len(RFI_flag.sk1)):
        if RFI_flag.flag_sk[j] == c[i]:
            skflag[j] = RFI_flag.sk1[i]
plt.plot(RFI_flag.flag_sk, skflag,'ro')
plt.savefig('freq-SK Plot_%s.png' %str(comm_rank))

fig3 = plt.figure()
ax = fig3.add_subplot(121)#freq -- Power Plot, origin
im = ax.imshow(origin)
plt.colorbar(im)
plt.xlabel('origin')
forceAspect(ax,aspect=1)

ax = fig3.add_subplot(122)#freq -- Power Plot,flagging
im=ax.imshow(flagged)
plt.colorbar(im)
plt.xlabel('RFI')
forceAspect(ax,aspect=1)
plt.savefig('freq-Power Plot_%s.png' %str(comm_rank))

#plt.show()
print time.localtime()