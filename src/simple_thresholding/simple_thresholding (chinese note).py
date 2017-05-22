#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
#three things haven't been made sure: the value of bins, y_limit, and the fitting model(Ae^(-ax^2))

data_file = '../samplerfi.dat'
#import data
origin = np.loadtxt(data_file)
row = origin.shape[0]
column = origin.shape[1]
RFI = ['Row', 'Colunm','Power' ]
output=open('Thresholding_RFI_Flag.txt','w')
output.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
flagged = np.loadtxt(data_file)

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(origin)-np.min(origin))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(origin), np.max(origin), step)

for i in np.arange(row):
    for j in np.arange(column):
        y_num = int((origin[i,j] - np.min(origin)) / step)
        if y_num == bins:
            y_hist[bins-1] = y_hist[bins-1] + 1
        else:
            y_hist[y_num] = y_hist[y_num] + 1　　　#y坐标是点的个数

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
for i in np.arange(row):
    for j in np.arange(column):
        if origin[i,j] >= x_limit:
            RFI.append([i,j,origin[i,j]])
            output.write(str(i) + '  ' + str(j) + '  ' + str(origin[i,j]) + '\n')
            flagged[i,j] = origin[i,j] + 1000  #make RFI more clear  标记的内容加上１０００，不标记的不加
output.close()
#print x_limit
#print RFI

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
