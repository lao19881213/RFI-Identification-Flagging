import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import scipy
from scipy.stats import kurtosis
from sympy import integrate,oo,exp,symbols,pi
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from scipy.optimize import curve_fit
from scipy.special import gamma
#note: this method is not very good! It doesn't work very well for the row RFI
origin=np.loadtxt('samplerfi.dat')
origin1=np.loadtxt('data_0.dat')
flagged=np.loadtxt('data_0.dat')

output = open('shuju.dat','w')
for i in np.arange(32):
    for j in np.arange(500):
        output.write(str(origin[i,j]) + '    ')
    output.write('\n')    
output.close()

origin2 = np.loadtxt('shuju.dat')

sk=scipy.stats.kurtosis(origin2,axis=0,fisher=False,bias=False) #0 is along freq, 1 is along time ,nan_policy='propagate'


#make histogram
bins = 100   #if bins is smaller, than y_limit must be larger
y_hist, bin_edge = np.histogram(sk, bins = bins) #this can be used, but the len(y_hist) = len(x_hist) + 1
x_hist = np.delete(bin_edge, len(bin_edge)-1)   #delete  the  last  number
#print len(x_hist)    100       bin_edge  shi  mei  fen  de  shu  ju
hgap = (x_hist[1]-x_hist[0]) / 2
x_hist = [i+hgap for i in x_hist]
#print bin_edge
print len(y_hist)
def curve_model(x, y0=1,A=1,m=1,v=1,alpha=1,lam=1):#Pearson Type IV distribution
    k = (2 ** (2 * m - 2)) * ((abs(gamma(complex(m, v / 2)))) ** 2) / (np.pi * alpha * gamma(2 * m - 1))
    #print 'k=',k
    func = y0+A*k*((1+(x-lam)/alpha)**(-m))*(np.exp(-v*np.arctan((x-lam)/alpha)))
    return func

CurveModel = custom_model(curve_model)#fit_deriv = curve_deriv)

#fitting the histogram with defined function
h_init = CurveModel()
fit_h = fitting.LevMarLSQFitter()
h = fit_h(h_init, x_hist, y_hist)

#plot
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

#
#fig1 = plt.figure()#histogram -- Number Plot, & Pearson 4 fitting
#plt.hist(sk,bins)
#plt.plot(x_hist, h(x_hist), label = 'fitting',color = 'red')

fig2 = plt.figure()#freq -- SK Plot
c=np.zeros(len(sk))
for i in range(len(sk)):
    c[i]=i
plt.plot(c, sk, 'yo')
skflag = np.zeros(len(flag_sk))
for j in np.arange(len(flag_sk)):
    for i in np.arange(len(sk)):
        if flag_sk[j] == c[i]:
            skflag[j] = sk[i]
plt.plot(flag_sk, skflag,'ro')


#fig3 = plt.figure()
#ax = fig3.add_subplot(121)#freq -- Power Plot, origin
#im = ax.imshow(origin)
#plt.colorbar(im)
#plt.xlabel('origin')
#forceAspect(ax,aspect=1)
#
#ax = fig3.add_subplot(122)#freq -- Power Plot,flagging
#im=ax.imshow(flagged)
#plt.colorbar(im)
#plt.xlabel('RFI')
#forceAspect(ax,aspect=1)