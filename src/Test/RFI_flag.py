#!/usr/bin/env python
import numpy as np
from numpy import *
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.convolution import Gaussian2DKernel, convolve_fft
import scipy
from scipy.stats import kurtosis
from sympy import integrate,oo,exp,symbols,pi
from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy import linalg
#three things haven'

class RFI_flag:
    #This method marks RFI by setting the threshold of the valve,and, in order to visualize, add the RFI data point value to 1000 
    def thresh(self,origin,local_origin,local_data_offset,comm_rank,row,column,RFI,bins,step,y_hist, x_hist,flagged_th):
        

        output=open('Thresholding_RFI_Flag_test_{}.txt'.format(comm_rank),'w')
        output.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
        
#        a = []
#        for i in np.arange(local_data_offset[1]):
#            a.append(np.min(local_origin[local_data_offset[comm_rank]+i,:]))
        
        for i in np.arange(row):
            for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
                y_num = int((local_origin[i,j] - np.min(origin)) / step)
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
        RFI_flag.h = fit_h(h_init, x_hist, y_hist)
        h1 = fit_h(h_init, x_hist1, y_hist1)
        
        #flag the RFI
        y_limit = 0.001   #have tried and find out 0.01 works better(for bins=10000) than 0.001
        x_limit = np.sqrt(-(np.log(y_limit/h1.A))/h1.a)
        for i in np.arange(row):
            for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
                if local_origin[i,j] >= x_limit:
                    RFI.append([i,j,local_origin[i,j]])
                    output.write(str(i) + '  ' + str(j) + '  ' + str(local_origin[i,j]) + '\n')
                    flagged_th[i,j] = local_origin[i,j] + 1000  #make RFI more clear
        output.close()
#        return x_hist,y_hist,flagged_th 
    
    
    
    #The method is to mark the RFI with the original data by gaussian filter  and  setting  the thresholding of value
    def gaus_filter(self,filtered,local_filtered,local_data_offset,comm_rank,row,column,RFI,bins,x_hist,y_hist,step,flagged_gf):
        output2 = open('2DGaussianFiltering_RFI_Flag_{}.txt'.format(comm_rank),'w')
        output2.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
        
#        a = []
#        for i in np.arange(local_data_offset[1]):
#            a.append(np.min(local_filtered[local_data_offset[comm_rank]+i,:]))
        
        for i in np.arange(row):
            for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
                y_num = int((local_filtered[i,j] - np.min(filtered)) / step)
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
        RFI_flag.h = fit_h(h_init, x_hist, y_hist)
        h1 = fit_h(h_init, x_hist1, y_hist1)
        
        #flag the RFI
        y_limit = 0.001
        x_limit = np.sqrt(-(np.log(y_limit/h1.A))/h1.a)
        for i in np.arange(row):
            for j in np.arange(local_data_offset[comm_rank],local_data_offset[comm_rank + 1]):
                if local_filtered[i,j] >= x_limit:
                    RFI.append([i,j,local_filtered[i,j]])
                    output2.write(str(i) + '  ' + str(j) + '  ' + str(local_filtered[i,j]) + '\n')
                    flagged_gf[i,j] = local_filtered[i,j] + 1000  #make RFI more clear
        output2.close()
          
    
    sk1 = []
    x_hist1 = []
    h1 = []
    flag_sk = []
    #Find the corresponding x (that is, kurtosis), then find the frequency channel corresponding to the kurtosis, and then mark it as RFI 
    def spectral_kurtosis(self,origin,local_origin,local_data_offset,comm_rank,column,row,bins,flagged):
        
        RFI_flag.sk1=scipy.stats.kurtosis(local_origin,axis=0,fisher=False,bias=False) #0 is along freq, 1 is along time ,nan_policy='propagate'
        #make histogram
        y_hist1, bin_edge1 = np.histogram(RFI_flag.sk1, bins = bins) #this can be used, but the len(y_hist) = len(x_hist) + 1
        RFI_flag.x_hist1 = np.delete(bin_edge1, len(bin_edge1)-1)
        hgap1 = (RFI_flag.x_hist1[1]-RFI_flag.x_hist1[0]) / 2
        RFI_flag.x_hist1 = [i+hgap1 for i in RFI_flag.x_hist1]
        
        
        sk=scipy.stats.kurtosis(origin,axis=0,fisher=False,bias=False) #0 is along freq, 1 is along time ,nan_policy='propagate'
        #make histogram
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
        RFI_flag.h1 = fit_h(h_init, RFI_flag.x_hist1, y_hist1)
        
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
                        RFI_flag.flag_sk.append(i)
                if sk[i] >= bin_edge[flag_hist[j][0]] and sk[i] < bin_edge[flag_hist[j][0]+1]:#flag in freq -- sk plot
                    a = RFI_flag.flag_sk
                    a.append(i)
        
        mean = np.mean(origin)
        median = np.median(origin)
        RFI = ['Row', 'Colunm','Power' ]
        output=open('SK_RFI_Flag_{}.txt'.format(comm_rank),'w')
        output.write(str(RFI[0])+'  '+str(RFI[1])+'  '+str(RFI[2])+'\n')
        for i in np.arange(len(RFI_flag.flag_sk)):
            l = [x[RFI_flag.flag_sk[i]] for x in local_origin]
            for row in np.arange(origin.shape[0]):
                if l[row] > median + 2:#flag in origin plot & output, here I just compare with the mean(or median) of all data
                    RFI.append([row, RFI_flag.flag_sk[i], local_origin[row, RFI_flag.flag_sk[i]]])
                    output.write(str(row) + '  ' + str(RFI_flag.flag_sk[i]) + '  ' + str(local_origin[row, RFI_flag.flag_sk[i]]) + '\n')
                    flagged[row,RFI_flag.flag_sk[i]] = local_origin[row, RFI_flag.flag_sk[i]] + 1000 #also, make RFI more clear.If plus 10, you can see the row RFI very clear, but it is not flagged!
#        return sk1,x_hist1

    #The method is to simply view the entire data as a 500-by-128 matrix, and do a singular value decomposition for it. 
    #Find the singular value, keep the value of the matrix with the maximum value, and the other singular values are 
    #set to 0, and you can go to noise.         
    def singular_value_decomposition(self,origin,local_origin,comm_rank):
        U, s, Vh = linalg.svd(origin)#, full_matrices = False)
        U1, s1, Vh1 = linalg.svd(local_origin)#, full_matrices = False)
        #print s1
       
        s_fix = np.zeros(len(s1))
        mean = np.mean(s)
        median = np.median(s)
        for i in range(len(s1)):
            if s1[i] > median:
                s_fix[i] = s1[i]
        S1 = linalg.diagsvd(s_fix, U1.shape[1], Vh1.shape[0])
        RFI_flag.SVD = np.dot(U1, np.dot(S1, Vh1))
        np.savetxt("SVD_{}.dat".format(comm_rank),RFI_flag.SVD)
#        print s1.shape