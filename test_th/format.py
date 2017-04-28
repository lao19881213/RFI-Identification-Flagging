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

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(origin)-np.min(origin))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(origin), np.max(origin), step)


for i in np.arange(128):
    for j in np.arange(500):
        y_num = int((origin[i,j] - np.min(origin)) / step)
        if y_num == bins:
            y_hist[bins-1] = y_hist[bins-1] + 1
        else:
            y_hist[y_num] = y_hist[y_num] + 1
#print np.max(y_hist)
##for i in np.arange(10000):
##    if y_hist[i] == 2693:
##        print i
#print y_hist[3001]
mean = np.mean(y_hist)            
print mean

def plt_1(x,y):
    bx=plt.gca()
    bx.set_xlim(np.min(origin),np.max(origin) + 1)
    bx.set_ylim(np.min(y_hist),100)
    plt.plot(x, y, 'ko', label = 'origin')
#    plt.plot(x, h(x), label = 'fitting',color = 'red')
    plt.xlabel('Power')
    plt.ylabel('Number')
    plt.legend()
    #plt.savefig('rfi_%s.png' %str(comm_rank))
    #plt.show()

fig1 = plt.figure(comm_rank)
plt_1(x_hist,y_hist)