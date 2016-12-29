import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from astropy.convolution import Gaussian2DKernel, convolve_fft
from PIL import Image
from pylab import *

# read data
#RFI = array(Image.open('rfidataSnapshot.png').convert('L'))  #this line read the image
RFI = np.loadtxt('samplerfi.dat')
# Create kernel
g = Gaussian2DKernel(stddev=0.5)

# Convolve data
RFIFiltered = convolve_fft(RFI, g) #boundary='extend')

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

fig = plt.figure()

ax = fig.add_subplot(121)
cmap = mpl.cm.hot
im=ax.imshow(RFI, cmap=cmap)
plt.colorbar(im)
#plt.axis('off')
forceAspect(ax,aspect=1)

ax = fig.add_subplot(122)
cmap = mpl.cm.hot
im=ax.imshow(RFIFiltered, cmap=cmap)
plt.colorbar(im)
#plt.axis('off')
forceAspect(ax,aspect=1)

plt.show()