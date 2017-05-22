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
flagged=np.loadtxt('samplerfi.dat')
sk=scipy.stats.kurtosis(origin,axis=0,fisher=False,bias=False) #0 is along freq, 1 is along time ,nan_policy='propagate'

#make histogram
bins = 100   #if bins is smaller, than y_limit must be larger
y_hist, bin_edge = np.histogram(sk, bins = bins) #this can be used, but the len(y_hist) = len(x_hist) + 1
x_hist = np.delete(bin_edge, len(bin_edge)-1)
hgap = (x_hist[1]-x_hist[0]) / 2
x_hist = [i+hgap for i in x_hist]


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

#Here because m should > 0.5, alpha > 0, but the astropy program cannot set the parameter range, thus here we use curve_fit.
#popt, pcov = curve_fit(curve_model, x_hist, y_hist, bounds=([-np.inf,-np.inf,0.5,-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
#curve_fit fits very bad! Have to use astropy.
#The other package may also work, which is LMFIT. It needs to be installed by pip. Web:http://cars9.uchicago.edu/software/python/lmfit/intro.html

#note that if m <= 0.5, the fitting is unsuccessful
if h.m <= 0.5:
    print 'Warning! Wrong fitting!!!'

#flag the RFI
pearson = np.zeros(len(y_hist))
part = []
flag_hist = []
flag_sk = []
for i in np.arange(len(y_hist)):
    pearson[i] = curve_model(x = x_hist[i], y0=h.y0*1, A=h.A*1, m=h.m*1, v=h.v*1, alpha=h.alpha*1, lam= h.lam*1)
for i in np.arange(len(y_hist)):
    if x_hist[i] <= 5:
        part.append(pearson[i])
for i in np.arange(len(part)):
    if part[i] == np.min(part):
        minnum = i
for i in np.arange(len(y_hist)):
    if pearson[i] <=1 or i <= minnum:
        flag_hist.append([i,x_hist[i]]) # flag in histogram, fig 1

for i in np.arange(len(sk)):
    for j in np.arange(len(flag_hist)):
        if flag_hist[len(flag_hist)-1][0] == len(x_hist) - 1:#notice in np.histogram, the last bin contains the right limit
            if sk[i] == bin_edge[len(bin_edge)-1]:
                flag_sk.append(i)
        if sk[i] >= bin_edge[flag_hist[j][0]] and sk[i] < bin_edge[flag_hist[j][0]+1]:#flag in freq -- sk plot
            flag_sk.append(i)

mean = np.mean(origin)
median = np.median(origin)
RFI = ['Row', 'Colunm','Power' ]
output=open('SK_RFI_Flag.txt','w')
output.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
for i in np.arange(len(flag_sk)):
    l = [x[flag_sk[i]] for x in origin]
    for row in np.arange(origin.shape[0]):
        if l[row] > median + 2:#flag in origin plot & output, here I just compare with the mean(or median) of all data
            RFI.append([row, flag_sk[i], origin[row, flag_sk[i]]])
            output.write(str(row) + '  ' + str(flag_sk[i]) + '  ' + str(origin[row, flag_sk[i]]) + '\n')
            flagged[row,flag_sk[i]] = origin[row, flag_sk[i]] + 1000 #also, make RFI more clear.If plus 10, you can see the row RFI very clear, but it is not flagged!
print len(flag_sk)
#plot
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

fig1 = plt.figure()#histogram -- Number Plot, & Pearson 4 fitting
plt.hist(sk,bins)        #sk is  axis  x     
plt.plot(x_hist, h(x_hist), label = 'fitting',color = 'red')

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

plt.show()
