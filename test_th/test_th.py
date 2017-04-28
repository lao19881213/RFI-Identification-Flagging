#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mpi4py.MPI as MPI
import os
from numpy import *
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
#three things haven't been made sure: the value of bins, y_limit, and the fitting model(Ae^(-ax^2))

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#import data
def loadtxt_th(data_file):
    origin = np.loadtxt(data_file)
    return origin
    
origin = loadtxt_th('samplerfi.dat')
row = origin.shape[0]
column = origin.shape[1]
RFI = ['Row','Column','Power']

#output=open('data_{0}.dat'.format(comm_rank),'w')
 #divide the data to each processor
local_data_offset = np.linspace(0, row, comm_size + 1).astype('int')
   
  #get the local data which will be processed in this processor
#local_data = origin[local_data_offset[comm_rank] :local_data_offset[comm_rank + 1]]
#print "****** %d/%d processor gets local data ****" %(comm_rank, comm_size)
#print local_data,comm_rank

data1 = zeros([row,column])
for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        data1[i,j] = origin[i,j]
np.savetxt('data_{}.dat'.format(comm_rank),data1)


local_origin = np.loadtxt('data_{}.dat'.format(comm_rank))


#local_origin = np.loadtxt('data_{0}.dat'.format(str(comm_rank)))
output=open('Thresholding_RFI_Flag_test_{}.txt'.format(comm_rank),'w')
output.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
flagged = np.loadtxt('data_{}.dat'.format(comm_rank))

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(origin)-np.min(origin))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(origin), np.max(origin), step)

a = []
for i in np.arange(local_data_offset[1]):
    a.append(np.min(local_origin[local_data_offset[comm_rank]+i,:]))

for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        y_num = int((local_origin[i,j] - np.min(a)) / step)
        if y_num == bins:
            y_hist[bins-1] = y_hist[bins-1] + 1
        else:
            y_hist[y_num] = y_hist[y_num] + 1
#print np.max(y_hist) 
      
y_hist1 = np.zeros(bins)
x_hist1 = np.arange(np.min(origin), np.max(origin), step)
for i in np.arange(row):
    for j in np.arange(column):
        y_num1 = int((origin[i,j] - np.min(origin)) / step)
        if y_num1 == bins:
            y_hist1[bins-1] = y_hist1[bins-1] + 1
        else:
            y_hist1[y_num1] = y_hist1[y_num1] + 1            
                      
#define fitting function
def curve_model(x, a = 1, A = 1):
    func = A * np.exp(- a * x)
    return func
#def curve_deriv(x,a = 1, A = 1):
#    d_a = a / 10
#    d_A = A / 10
#    return [d_a,d_A]
CurveModel = custom_model(curve_model)#fit_deriv = curve_deriv)

#fitting the histogram with defined function
h_init = CurveModel()
fit_h = fitting.LevMarLSQFitter()
h = fit_h(h_init, x_hist, y_hist)
h1 = fit_h(h_init, x_hist1, y_hist1)

#flag the RFI
y_limit = 0.001   #have tried and find out 0.01 works better(for bins=10000) than 0.001
x_limit = np.sqrt(-(np.log(y_limit/h1.A))/h1.a)
for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        if local_origin[i,j] >= x_limit:
            RFI.append([i,j,local_origin[i,j]])
            output.write(str(i) + '  ' + str(j) + '  ' + str(local_origin[i,j]) + '\n')
            flagged[i,j] = local_origin[i,j] + 1000  #make RFI more clear
output.close()

#plot the histogram fitting & data RFI
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def plt_1(x,y):
    bx=plt.gca()
    bx.set_xlim(np.min(origin),np.max(origin) + 1)
    bx.set_ylim(np.min(y_hist),100)
    plt.plot(x, y, 'ko', label = 'origin')
    plt.plot(x, h(x), label = 'fitting',color = 'red')
    plt.xlabel('Power')
    plt.ylabel('Number')
    plt.legend()
    #plt.savefig('rfi_%s.png' %str(comm_rank))
    #plt.show()

fig1 = plt.figure(comm_rank)
plt_1(x_hist,y_hist)

def plt_2(m,n):             #ax = m ,im = n
    plt.colorbar(n)#, cax)
    plt.xlabel('origin')
    forceAspect(m,aspect=1)
    
def plt_3(m,n):
    plt.colorbar(n)
    plt.xlabel('RFI')
    forceAspect(m,aspect=1)
    
fig2 = plt.figure()
ax= fig2.add_subplot(121)
im=ax.imshow(origin)
plt_2(ax,im)

ax = fig2.add_subplot(122)
im=ax.imshow(flagged)
plt_3(ax,im)    
#plt.savefig('RFI_%s.png' %str(comm_rank))  
plt.show()
