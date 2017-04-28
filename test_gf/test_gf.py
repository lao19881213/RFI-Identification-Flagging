#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI
#import matplotlib.image as mpimg
from astropy.convolution import Gaussian2DKernel, convolve_fft
from PIL import Image
from pylab import *
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def loadtxt_gf(data_file):
    origin = np.loadtxt(data_file)
    return origin
    
origin = loadtxt_gf('samplerfi.dat')
# 2D Gaussian Filtering
#RFI = array(Image.open('rfidataSnapshot.png').convert('L'))  #this line read the image data
#origin = np.loadtxt(data_file)
f = Gaussian2DKernel(stddev=0.5)
filtered = convolve_fft(origin, f) #boundary='extend')
row = filtered.shape[0]
column = filtered.shape[1]
np.savetxt('2DGaussianFiltering.dat',filtered)

#divide the data to each processor
local_data_offset = np.linspace(0, row, comm_size + 1).astype('int')
   
#get the local data which will be processed in this processor
local_data = filtered[local_data_offset[comm_rank] :local_data_offset[comm_rank + 1]]
#print "****** %d/%d processor gets local data ****" %(comm_rank, comm_size)
#print local_data,comm_rank

data1 = zeros([row,column])
for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        data1[i,j] = filtered[i,j]
np.savetxt('2DGaussianFiltering_{}.dat'.format(comm_rank),data1)

local_filtered = np.loadtxt('2DGaussianFiltering_{}.dat'.format(comm_rank))


#prepare for filtered data and output
RFI = ['Row', 'Colunm','Power' ]
output2 = open('2DGaussianFiltering_RFI_Flag_{}.txt'.format(comm_rank),'w')
output2.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
flagged = convolve_fft(local_filtered, f)

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(filtered)-np.min(filtered))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(filtered), np.max(filtered), step)

a = []
for i in np.arange(local_data_offset[1]):
    a.append(np.min(local_filtered[local_data_offset[comm_rank]+i,:]))

for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        y_num = int((local_filtered[i,j] - np.min(a)) / step)
        if y_num == bins:
            y_hist[bins-1] = y_hist[bins-1] + 1
        else:
            y_hist[y_num] = y_hist[y_num] + 1

y_hist1 = np.zeros(bins)
x_hist1 = np.arange(np.min(filtered), np.max(filtered), step)
for i in np.arange(row):
    for j in np.arange(column):
        y_num1 = int((filtered[i,j] - np.min(filtered)) / step)
        if y_num1 == bins:
            y_hist1[bins-1] = y_hist1[bins-1] + 1
        else:
            y_hist1[y_num1] = y_hist1[y_num1] + 1

#define fitting function
def curve_model(x, a = 1, A = 1):
    func = A * np.exp(- a * x)
    return func
CurveModel = custom_model(curve_model)

#fitting the histogram with defined function
h_init = CurveModel()
fit_h = fitting.LevMarLSQFitter()
h = fit_h(h_init, x_hist, y_hist)
h1 = fit_h(h_init, x_hist1, y_hist1)

#flag the RFI
y_limit = 0.001
x_limit = np.sqrt(-(np.log(y_limit/h1.A))/h1.a)
for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        if local_filtered[i,j] >= x_limit:
            RFI.append([i,j,local_filtered[i,j]])
            output2.write(str(i) + '  ' + str(j) + '  ' + str(local_filtered[i,j]) + '\n')
            flagged[i,j] = local_filtered[i,j] + 1000  #make RFI more clear
output2.close()
#print x_limit
#print RFI

#plot the histogram fitting & data RFI
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def plt_1(x,y):
    bx=plt.gca()
    bx.set_xlim(np.min(origin),np.max(origin))
    bx.set_ylim(0,100)
    plt.plot(x, y, 'k.', label = 'filtered')
    plt.plot(x, h(x), label = 'fitting',color = 'red')
    plt.xlabel('Power')
    plt.ylabel('Number')
    plt.legend()
    
fig1 = plt.figure(comm_rank)
plt_1(x_hist,y_hist)
#plt.savefig('rfi_%s.png' %str(comm_rank))

def plt_2(m,n):
    plt.colorbar(n)
    plt.xlabel('origin')
    forceAspect(m,aspect=1)
    
def plt_3(m,n):
    plt.colorbar(n)
    plt.xlabel('filtered')
    forceAspect(m,aspect=1)
    
def plt_4(m,n):
    plt.colorbar(n)
    plt.xlabel('RFI flagged')
    forceAspect(m,aspect=1)    
    
fig2 = plt.figure()
ax = fig2.add_subplot(131)
im=ax.imshow(origin)
plt_2(ax,im)

ax = fig2.add_subplot(132)
im=ax.imshow(filtered)
plt_3(ax,im)

ax = fig2.add_subplot(133)
im=ax.imshow(flagged)
plt_4(ax,im)

plt.show()
#plt.savefig('RFI_%s.png' %str(comm_rank))
