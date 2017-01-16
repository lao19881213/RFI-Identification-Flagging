#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from astropy.convolution import Gaussian2DKernel, convolve_fft
from PIL import Image
from pylab import *
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model

data_file = '../samplerfi.dat'
# 2D Gaussian Filtering
#RFI = array(Image.open('rfidataSnapshot.png').convert('L'))  #this line read the image data
origin = np.loadtxt(data_file)
f = Gaussian2DKernel(stddev=0.5)
filtered = convolve_fft(origin, f) #boundary='extend')

#output filtered data
output1 = open('2DGaussianFiltering.txt','w')
for i in filtered:
    k = ' '.join([str(j) for j in i])
    output1.write(k + "\n")
output1.close()

#prepare for filtered data and output
row = filtered.shape[0]
column = filtered.shape[1]
RFI = ['Row', 'Colunm','Power' ]
output2 = open('2DGaussianFiltering_RFI_Flag.txt','w')
output2.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
flagged = convolve_fft(origin, f)

#make histogram
bins = 10000   #if bins is smaller, than y_limit must be larger
step = (np.max(filtered)-np.min(filtered))/ bins #seperate into 100 bins
y_hist = np.zeros(bins)
x_hist = np.arange(np.min(filtered), np.max(filtered), step)

for i in np.arange(row):
    for j in np.arange(column):
        y_num = int((filtered[i,j] - np.min(filtered)) / step)
        if y_num == bins:
            y_hist[bins-1] = y_hist[bins-1] + 1
        else:
            y_hist[y_num] = y_hist[y_num] + 1

#define fitting function
def curve_model(x, a = 1, A = 1):
    func = A * np.exp(- a * x)
    return func
CurveModel = custom_model(curve_model)

#fitting the histogram with defined function
h_init = CurveModel()
fit_h = fitting.LevMarLSQFitter()
h = fit_h(h_init, x_hist, y_hist)

#flag the RFI
y_limit = 0.001
x_limit = np.sqrt(-(np.log(y_limit/h.A))/h.a)
for i in np.arange(row):
    for j in np.arange(column):
        if filtered[i,j] >= x_limit:
            RFI.append([i,j,filtered[i,j]])
            output2.write(str(i) + '  ' + str(j) + '  ' + str(filtered[i,j]) + '\n')
            flagged[i,j] = filtered[i,j] + 1000  #make RFI more clear
output2.close()
print x_limit
#print RFI

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
plt.plot(x_hist, h(x_hist), label = 'fitting',color = 'red')
plt.xlabel('Power')
plt.ylabel('Number')

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
im=ax.imshow(flagged)#, cmap=cmap)
plt.colorbar(im)
plt.xlabel('RFI flagged')
#plt.axis('off')
forceAspect(ax,aspect=1)

plt.show()
