#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpi4py.MPI as MPI
#three things haven't been made sure: the value of bins, y_limit, and the fitting model(Ae^(-ax^2))

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

data_file = 'samplerfi.dat'
#import data
origin = np.loadtxt(data_file)
row = origin.shape[0]
column = origin.shape[1]
RFI = ['Row', 'Column','Power' ]

#if comm_rank == 0:
  #  all_data = np.arange(20).reshape(4, 5)
 #   print "************ data ******************"
    #print origin
   
    #broadcast the data to all processors
origin = comm.bcast(origin if comm_rank == 0 else None, root = 0)
   
    #divide the data to each processor
local_data_offset = np.linspace(0, row, comm_size + 1).astype('int')
   
   #get the local data which will be processed in this processor
local_data = origin[local_data_offset[comm_rank] :local_data_offset[comm_rank + 1]]
print "****** %d/%d processor gets local data ****" %(comm_rank, comm_size)
print local_data,comm_rank

output=open('Thresholding_RFI_Flag_test2.txt','w')
output.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
flagged = np.loadtxt(data_file)

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(origin)-np.min(origin))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(origin), np.max(origin), step)

for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        y_num = int((local_data[i,j] - np.min(local_data)) / step)
        if y_num == bins:
            y_hist[bins-1] = y_hist[bins-1] + 1
        else:
            y_hist[y_num] = y_hist[y_num] + 1

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

#flag the RFI
y_limit = 0.001   #have tried and find out 0.01 works better(for bins=10000) than 0.001
x_limit = np.sqrt(-(np.log(y_limit/h.A))/h.a)
for i in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
    for j in np.arange(column):
        if local_data[i,j] >= x_limit:
            RFI.append([i,j,local_data[i,j]])
            output.write(str(i) + '  ' + str(j) + '  ' + str(local_data[i,j]) + '\n')
            flagged[i,j] = local_data[i,j] + 1000  #make RFI more clear
output.close()

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
plt.plot(x_hist, h(x_hist), label = 'fitting',color = 'red')
plt.xlabel('Power')
plt.ylabel('Number')
plt.legend()

fig2 = plt.figure()
ax = fig2.add_subplot(121)
#cmap = mpl.cm.hot
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
im=ax.imshow(origin)#, cmap=cmap)
plt.colorbar(im)#, cax)
plt.xlabel('origin')
#plt.axis('off')
forceAspect(ax,aspect=1)

ax = fig2.add_subplot(122)
#cmap = mpl.cm.hot
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
im=ax.imshow(flagged)#, cmap=cmap)
plt.colorbar(im)
plt.xlabel('RFI')
#plt.axis('off')
forceAspect(ax,aspect=1)

plt.show()
