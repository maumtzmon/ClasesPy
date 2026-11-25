
import numpy as np
from scipy.stats import poisson, norm, chisquare, geom
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
import glob
import numpy as np
import numpy.ma as ma
from skimage import feature, measure
from astropy.io import fits
import multiprocessing as mp
from joblib import Parallel, delayed,cpu_count
import pickle as pkl
from scipy.optimize import curve_fit, OptimizeWarning
from dateutil.parser import parse
import datetime
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import binary_dilation

osc=550
active_mask = np.s_[:, 0:osc]
os_median_mask = np.s_[:, osc:]
overscan_mask = np.s_[5:-5, -(osc-5):-5]


def chi_sqaured_value(class_marks,histogram,function,*popt):
    fit = function(class_marks,*popt)
    chi2 = np.sum(((histogram-fit)**2)/fit)
    dgf = len(histogram)-len(popt)-1 
    return chi2, dgf

def GPD_curvefit(x,*popt):
    a,p,lam,mu,sigma = popt
    nmax = 10 
    result = 0.0          
    for n in range(0,nmax,1):
        result += a*geometrical_poisson_distribution(n,p,lam)*norm.pdf(x,loc=mu+n,scale=sigma)
    
    return result

def normal_curvefit(x,*popt):
    a,mu,sigma = popt
    result = a*norm.pdf(x,loc=mu,scale=sigma)
    return result

def geometrical_poisson_distribution(n,*popt):
    p,lam = popt        
    factor_exp = np.exp(-lam)
    if type(n) is int:
        ni = n
        vector_factor = np.zeros([ni])
        for k in range(1,ni+1,1):
            if k == 1:
                val = lam*((1-p)**(ni-1))*p
                # vector_geom[k-1] = (p**(n-k))*((1-p)**(k))
                vector_factor[k-1] = val
                # print(val)

            else:
                val *= (lam*(p/(1-p)))*((ni-k)/(k*(k-1)))
                # val *= (((p))*(factor_exp*lam)*(n-1))/((k-1)*k*(1-p))
                # vector_geom[k-1] = p**(n-k)*(1-p)**(k)
                vector_factor[k-1] = val
                # print(val)

        if len(vector_factor) == 0:
            vector_prob = factor_exp
        else:
            vector_prob = factor_exp*np.sum(vector_factor)

    else:
        nmax = int(len(n))
        vector_prob = np.zeros([nmax])

        for ni in range(1,nmax+1,1):
            vector_factor = np.zeros([ni])
            for k in range(1,ni+1,1):
                if k == 1:
                    val = lam*((1-p)**(ni-1))*p
                    # vector_geom[k-1] = (p**(n-k))*((1-p)**(k))
                    vector_factor[k-1] = val
                    # print(val)

                else:
                    val *= (lam*(p/(1-p)))*((ni-k)/(k*(k-1)))
                    # val *= (((p))*(factor_exp*lam)*(n-1))/((k-1)*k*(1-p))
                    # vector_geom[k-1] = p**(n-k)*(1-p)**(k)
                    vector_factor[k-1] = val
                    # print(val)

        if len(vector_factor) == 0:
            vector_prob[ni-1] = factor_exp
        else:
            vector_prob[ni-1] = factor_exp*np.sum(vector_factor)
    
    return vector_prob

def makeHistogram(data, range=None, density=True):
    if range == None:
        histo, bins_edges = np.histogram(data,bins='fd',density=density)
    else:
        histo, bins_edges = np.histogram(data,bins='fd',range=range,density=density)
    class_marks = (bins_edges[:-1]+bins_edges[1:])/2

    return histo, bins_edges, class_marks


def Plot2Images(Image,Mask,MinRange=None,MaxRange=None, colorBar_1=True, colorBar_2=True):
    # fig, (ax1,ax2) = plt.subplots(1,2)
    try:
        if MinRange > MaxRange:
            Aux = MaxRange
            MaxRange = MinRange
            MinRange = Aux
    except TypeError:
        MinRange=None
        MaxRange=None
    # ax1.imshow(Mask, cmap='viridis',origin="lower")
    # ax2.imshow(Image, cmap='viridis',vmin=MinRange,vmax=MaxRange,origin="lower")

    fig=plt.figure(figsize=(20,10))

    plt.subplot(1,2,2)
    
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')

    if Mask.max()==1:
        im=plt.imshow(Mask, cmap='viridis',vmin=Mask.min(),vmax=Mask.max(), origin="lower")
    else:
        im=plt.imshow(Mask, cmap='viridis',vmin=MinRange,vmax=MaxRange, origin="lower")
    if colorBar_1:
        cbar=fig.colorbar(im)
    #cbar.set_label('boolean')

    plt.subplot(1,2,1)
    # plt.title(Image)  
    plt.ylabel('Y_pix')
    plt.xlabel('X_pix')
    im=plt.imshow(Image, cmap='viridis', vmin=MinRange,vmax=MaxRange,origin="lower")
    if colorBar_2:
        cbar=fig.colorbar(im)
    #cbar.set_label('[e-]')

    
    plt.show()


def gauss(x,x0,s,A):
    return A*np.exp(-(x-x0)**2/(2*s**2))/np.sqrt(np.pi*2*s**2)

def ax_nob(x,a):
    return a*x

def LocalSigma(data,extensions=1):
    if extensions==1:
        oscam_hist, bins = np.histogram( data[overscan_mask].flatten(), bins = np.arange(-.5,.5,0.01) )
        x = (bins[1:] + bins[:-1])/2
        try:
            popt, pcov = curve_fit( gauss, x, oscam_hist,
                                   p0=[ 0,0.2, np.sum(oscam_hist)])
        except (RuntimeError,OptimizeWarning,RuntimeWarning):
            print("Error - readout noise fit failed")

        sigma = [abs(popt[1])]
        perr = np.sqrt(np.diag(pcov))
        sigma_err = [perr[1]]
        
    else:
        sigma=[]
        sigma_err=[]
        for i in range(extensions):
            oscam_hist, bins = np.histogram( data[i][overscan_mask].flatten(), bins = np.arange(-.5,.5,0.01) )
            x = (bins[1:] + bins[:-1])/2
            try:
                popt, pcov = curve_fit( gauss, x, oscam_hist,
                                       p0=[ 0,0.2, np.sum(oscam_hist)])
                sigma_list = abs(popt[1]) 
                perr = np.sqrt(np.diag(pcov))
                sigma_err_list = perr[1]
            except (RuntimeError,OptimizeWarning,RuntimeWarning):
                print("Error - readout noise fit failed")

                sigma_list = -1000
                sigma_err_list = -1000
            
            sigma.append(sigma_list)
            sigma_err.append(sigma_err_list)
            
    return sigma, sigma_err


def precal(hdu_list,extensions=1,chid=0):
     #------------------------ subtract overscan medians
    if extensions==1:
        data = hdu_list[chid].data
        data -= np.median( data[os_median_mask], axis=1, keepdims=True )  
        
    else:
        data_list=[]
        for i in range(extensions):
            data = hdu_list[i].data
            data -= np.median( data[os_median_mask], axis=1, keepdims=True )  
            data_list.append(data)
        data_ac=np.array(data_list)

       #--------------------- from Pedro - cleaning Xtalk
        data=np.zeros(data_ac.shape)
        for i in range(extensions):
            data[i] = data_ac[i]
            for j in range(extensions):
                if(i != j):
                    try:
                        cut = (data_ac[i] > -1000) & (data_ac[i] < 0) & (data_ac[j] > 1e6)  # cut
                        if len(data_ac[j][cut].flatten())!=0:
                            popt, pcov = curve_fit(ax_nob, data_ac[i][cut], data_ac[j][cut])
                            data[i] -= data_ac[j]/popt
                    except (RuntimeError,OptimizeWarning,RuntimeWarning):
                        print("Error - Xtalk fit failed")
                   
    return data


def GetRDtime(hdu_list):

    start=parse(hdu_list[0].header["DATESTART"]+"Z").timestamp()
    end=parse(hdu_list[0].header["DATEEND"]+"Z").timestamp()
    time=datetime.timedelta(seconds=end-start)
    readout_time=time.total_seconds()/86400
    return readout_time

def gaussian2(x,m,s,g,a1,a2):
    return a1*np.exp(-1/2*((x-m)/s)**2)+a2*np.exp(-1/2*((x-m-g)/s)**2)

def maskedStatistics(npArray,n = 4,mask=None,tol=0.1):
    """npArray -- Contains imaga data as numpy array,\n
    n = 5 -- number of standard deviation,\n 
    mask -- a previous mask to be considered previously to computation,\\
    tol = 0.1 -- Tolerance to convergence
    """
    mean = float(np.ma.average(npArray))
    stdDev = float(np.ma.std(npArray))
    Count = 0
    MaxIter = 20
    flag = True
    meanPre = mean
    if np.any(mask) != None:
        Mask = mask
    while (Count < MaxIter) and (flag == True):
        Lrange = mean - n * stdDev
        Hrange = mean + n * stdDev
        Mask = np.logical_or(np.less_equal(npArray,Lrange), np.greater_equal(npArray,Hrange))
        npArrayMasked = np.ma.array(npArray, mask=Mask)
        mean = float(np.ma.average(npArrayMasked))
        stdDev = float(np.ma.std(npArrayMasked))
        Count += 1

        if np.abs(mean - meanPre) < tol:
            flag = False
        else:
            meanPre = mean
    return mean, stdDev, npArrayMasked, Mask

def LocalCalib(data,extensions=1):
    if extensions==1:
        hist, bins = np.histogram(data[active_mask].flatten(), bins = np.arange(-400, 1000, 1) )
        Mean, StdDev, _, _ = maskedStatistics(data[active_mask], n = 4)
        x = (bins[1:] + bins[:-1])/2
        try:
            popt_g, pcov = curve_fit(gaussian2, x, hist, p0=[ 0, 60, 200, 1e3, 1e2 ] )
            # popt_g, pcov = curve_fit(normal_poisson_distribution_3, x, hist,bounds=([0,-100,0,0,0,0,100],[1e6,100,200,1e5,1e5,1e5,300]), p0=[hist.max(),Mean,StdDev,hist.max()/2,hist.max()/4,hist.max()/8,200])
            
        except (RuntimeError,OptimizeWarning,RuntimeWarning):
            print(f"Error - gain fit failed")
        perr = np.sqrt(np.diag(pcov))
        gain = abs(popt_g[2])
        gain_err = perr[2]
        data=(data)/gain
        data_c = np.copy(data)
    else:
        gain=[]
        gain_err=[]
        for i in range(extensions):
            hist, bins = np.histogram( data[i][active_mask].flatten(), bins = np.arange(-400, 1000, 1) )
            x = (bins[1:] + bins[:-1])/2
            Mean, StdDev, _, _ = maskedStatistics(data[i][active_mask], n = 4)
            try:
                # popt_g, pcov = curve_fit(normal_poisson_distribution_3, x, hist,bounds=([0,-100,0,0,0,0,100],[1e6,100,200,1e5,1e5,1e5,300]), p0=[hist.max(),Mean,StdDev,hist.max()/2,hist.max()/4,hist.max()/8,200])
                popt_g, pcov = curve_fit( gaussian2, x, hist, p0=[ 0, 60, 200, 1e3, 1e2] )
                                                            # 1x2 p0=[ 0, 150, 770, 1e3, 1e2 ]
                                                            # 1x10 p0=[ 0, 150, 770, 1e2, 1e0 ] 
                perr = np.sqrt(np.diag(pcov))
                gain_list = abs(popt_g[2])
                # gain_list = abs(popt_g[-1])
                gain_err_list = perr[2]

                data[i]=data[i]/gain_list
            except (RuntimeError,OptimizeWarning,RuntimeWarning):
                print(f"Error - gain fit failed")
                gain_list = -1000
                gain_err_list = -1000
            
            gain.append(gain_list)
            gain_err.append(gain_err_list)
            data_c = np.copy(data)
    return gain, gain_err, data_c


def PlotImage(Image,MinRange=None,MaxRange=None):
    try:
        if MinRange > MaxRange:
            Aux = MaxRange
            MaxRange = MinRange
            MinRange = Aux
    except TypeError:
        MinRange=None
        MaxRange=None
    plt.imshow(Image, cmap='viridis',vmin=MinRange,vmax=MaxRange,origin="lower")
    plt.colorbar()
    plt.show()

def writeHistogram(filename,class_marks,histogram):
    strhisto = ""
    for p,q in zip(class_marks,histogram):
        strhisto += "{},{}\n".format(p,q)
    
    file = open(filename,"w")
    file.write(strhisto)
    file.close()
    return 0

def probabilisticAnalysisSRE(array,number_of_pixels=7,threshold=1,windowSize=10,mu=None,sigma=None):
    x_size, y_size = array.shape
    number_of_pixels_threshold = number_of_pixels
    # mask = np.zeros((x_size,y_size),dtype=np.uint8)
    nonextended = np.zeros((x_size,y_size),dtype=np.uint8)
    mask2 = np.zeros((x_size,y_size),dtype=np.uint8)
    windowSize2 = 5 

    if mu == None or sigma == None:
        mu, sigma, npArrayMasked, Mask = maskedStatistics(array,n=4)

    CCDStats = [mu, sigma]
    # threshold = norm.isf(p,loc=mu,scale=sigma)
    # for ps in range(0,windowSize+1,1):
    #     if binom.pmf(ps,windowSize,p) < pth:
    #         number_of_pixels_threshold = ps
    #         break 
    # number_of_pixels_threshold += 1  
    OTarray = np.where(array >= threshold, 1,0)
    mask = horizontalSREsearch(OTarray,windowSize,number_of_pixels_threshold)

    # threshold = norm.isf(p*5,loc=mu,scale=sigma)
    OTarray = np.where(array >= threshold, 1,0)
    # number_of_frames2 = x_size // windowSize2
    # mask_index2 = np.arange(0,windowSize2,1)
    number_of_pixels_threshold2 = 3
    mask2 = verticalSREsearch(OTarray,windowSize2,number_of_pixels_threshold2)

    mask = np.logical_or(mask,mask2)

    # label=ndimage.label(mask,structure=[[1,1,1],[1,1,1],[1,1,1]])
    # label2=ndimage.label(mask2,structure=[[1,1,1],[1,1,1],[1,1,1]])

    # rps=sk.regionprops(label,intensity_image=mask,cache=False, extra_properties=[sum_intensity])

    serialRegisterEvents, gradCCDMask, nonextended = makeSREMaskGrad(mask)
    # fit_Convolution_Normal_Poisson(array,k=15)

    return serialRegisterEvents, CCDStats, nonextended


def horizontalSREsearch(array,windowSize,numberofpixels):
    x_size, y_size = array.shape
    mask = np.zeros((x_size,y_size),dtype=np.uint8)
    mask_index = np.arange(0,windowSize,1)
    number_of_frames = y_size // (windowSize/2)
    number_of_frames -= 2
    for ps in range(0,int(number_of_frames),1):
        indexmin = int(ps*(windowSize/2))
        indexmax = indexmin + windowSize
        sumArray = np.sum(array[:,indexmin:indexmax],axis=1)
        AS = array[:,indexmin:indexmax]*mask_index
        ki = 10000
        AS[AS==0]=ki
        indexminArray = np.min(AS, axis=1)
        AS[AS==ki]=-ki
        indexmaxArray = np.max(AS, axis=1)
        AS[AS==-ki]=0

        for qs in range(0,AS.shape[0],1):
            if sumArray[qs] >= numberofpixels:
                AS[qs,indexminArray[qs]:indexmaxArray[qs]+1] = 1
                # AS[qs,:] = 1
            else:
                AS[qs,:] = 0

        mask[:,indexmin:indexmax] = AS

    return mask

def verticalSREsearch(array,windowSize,numberofpixels):
    x_size, y_size = array.shape
    number_of_frames2 = x_size // windowSize
    mask = np.zeros((x_size,y_size),dtype=np.uint8)
    for ps in range(0,number_of_frames2,1):
        indexmin = ps*windowSize
        indexmax = (ps+1)*windowSize
        sumArray = np.sum(array[indexmin:indexmax,:],axis=0,dtype=np.uint8)
        AS = np.zeros([windowSize,y_size])
        # ki = 10000
        # AS[AS==0]=ki
        # indexminArray = np.min(AS, axis=0)
        # AS[AS==ki]=-ki
        # indexmaxArray = np.max(AS, axis=0)
        # AS[AS==-ki]=0

        for qs in range(0,AS.shape[1],1):
            if sumArray[qs] >= numberofpixels:
                # AS[qs,indexminArray[qs]:indexmaxArray[qs]+1] = 1
                AS[:,qs] = 1


        mask[indexmin:indexmax,:] = AS

    return mask


def makeSREMaskGrad(array,extend=True):
    """image -- numpy array which contains image,\n
    n = 5 -- number of standard deviations (SD),\n
    nOS = 5 -- number of SD used to analyze overscan region if present,\n
    extend=True -- Make Serial register events twice larger,\n
    overscan=None -- Considers overscan region to be analyzed independently,\n
    frames=10 -- Default number of frames to divide horizontally full image."""
    ylenaux, xlenaux = np.shape(array)
    row = np.zeros([2,xlenaux])
    arrayAux = np.copy(array)
    arrayAux = np.concatenate((row,arrayAux,row),axis=0)
    gradCCDMask = [np.gradient(arrayAux.astype('float'),axis=0,edge_order=1),np.gradient(arrayAux.astype('float'),axis=1,edge_order=1)]
    ylenaux, xlenaux = np.shape(arrayAux)
    serialRegisterEvents = np.zeros([ylenaux,xlenaux])
    for i in range(1,ylenaux-2,1):
        for j in range(1,xlenaux-2,1):
            centralPixel = [gradCCDMask[0][i,j],gradCCDMask[1][i,j]]
            upperupperPixel = [gradCCDMask[0][i+2,j],gradCCDMask[1][i+2,j]]
            upperPixel = [gradCCDMask[0][i+1,j],gradCCDMask[1][i+1,j]]
            lowerPixel = [gradCCDMask[0][i-1,j],gradCCDMask[1][i-1,j]]
            lowerlowerPixel = [gradCCDMask[0][i-2,j],gradCCDMask[1][i-2,j]]
            leftPixel = [gradCCDMask[0][i,j-1],gradCCDMask[1][i,j-1]]
            rightPixel = [gradCCDMask[0][i,j+1],gradCCDMask[1][i,j+1]]
            upperleftPixel = [gradCCDMask[0][i+1,j-1],gradCCDMask[1][i+1,j-1]]
            upperrightPixel = [gradCCDMask[0][i+1,j+1],gradCCDMask[1][i+1,j+1]]
            lowerleftPixel = [gradCCDMask[0][i-1,j-1],gradCCDMask[1][i-1,j-1]]
            lowerrightPixel = [gradCCDMask[0][i-1,j+1],gradCCDMask[1][i-1,j+1]]
            if centralPixel[0] == 0 and upperPixel[0] < 0 and lowerPixel[0] > 0 and  upperleftPixel[0] < 0 and upperrightPixel[0] < 0 and lowerleftPixel[0] > 0 and lowerrightPixel[0] > 0:
                if centralPixel[1] == 0 and upperPixel[1] == 0 and lowerPixel[1] == 0 and upperleftPixel[1] == 0 and upperrightPixel[1] == 0 and lowerleftPixel[1] == 0 and lowerrightPixel[1] == 0:
                    if centralPixel[0] == 0 and leftPixel[0] == 0 and rightPixel[0] == 0:
                        # if upperupperPixel[0] == 0 and lowerlowerPixel[0] == 0:
                        serialRegisterEvents[i,j] = 1
            
        
        # if overscanFlag and ps == totalFrames - 1:
        #     serialRegisterEvents[:,overscan:] = np.logical_or(serialRegisterEvents[:,overscan:],serialRegisterEventsAux)
        #     CCDMask[:,overscan:] = np.logical_or(CCDMask[:,overscan:],CCDMaskAux)
        
        # elif overscanFlag and ps < totalFrames - 1:
        #     serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),0:overscan] = np.logical_or(serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),0:overscan],serialRegisterEventsAux)
        #     CCDMask[((ps-1)*framelen):((ps+1)*framelen),0:overscan] = np.logical_or(CCDMask[((ps-1)*framelen):((ps+1)*framelen),0:overscan],CCDMaskAux)        
        # elif not(overscanFlag):
        #     serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),:] = np.logical_or(serialRegisterEvents[((ps-1)*framelen):((ps+1)*framelen),:],serialRegisterEventsAux)
        #     CCDMask[((ps-1)*framelen):((ps+1)*framelen),:] = np.logical_or(CCDMask[((ps-1)*framelen):((ps+1)*framelen),:],CCDMaskAux)        
    
    nonextended_serialRegisterEvents = np.copy(serialRegisterEvents)

    if extend:
        serialRegisterEvents = extendSRE(serialRegisterEvents)
                        
    return serialRegisterEvents[2:-2], gradCCDMask, nonextended_serialRegisterEvents[2:-2]


def extendSRE(serialRegisterEvents,minlength = 20):
    
    ylen, xlen = np.shape(serialRegisterEvents)

    for i in range(0,ylen-1,1):
            if np.any(serialRegisterEvents[i,:]):
                numberOfPixels = np.sum(serialRegisterEvents[i,:])
                if numberOfPixels == 1:
                    array = np.where(serialRegisterEvents[i,:] == 1,np.arange(0,xlen,1),np.inf)
                    xmin = np.min(array)
                    xmin = xmin - 10
                    xmax = xmin + 10
                    if xmin < 0:
                        xmin = 0
                    
                    if xmax > xlen:
                        xmax = xlen

                    serialRegisterEvents[i,int(xmin):int(xmax)] = 1
                    # serialRegisterEvents[i,:] = np.where(serialRegisterEvents[i,:] == 1,0,serialRegisterEvents[i,:])
                else:
                    array = np.where(serialRegisterEvents[i,:] == 1,np.arange(0,xlen,1),np.inf)
                    xmin = np.min(array)
                    array = np.where(serialRegisterEvents[i,:] == 1,np.arange(0,xlen,1),-np.inf)
                    xmax = np.max(array)

                    lengthEvent = xmax-xmin
                    if lengthEvent < minlength:
                        lengthEvent = minlength

                    lengthEvent = np.round(3*lengthEvent/4)

                    if xmin - lengthEvent < 0:
                        xmin = 0
                    else:
                        xmin = xmin - lengthEvent
                    
                    if xmax + lengthEvent > xlen - 1:
                        xmax = xlen
                    else:
                        xmax = xmax + lengthEvent

                    serialRegisterEvents[i,int(xmin):int(xmax)] = 1                              
    
    return serialRegisterEvents

def generateFITSFile(filename,data,headerL):
    image = fits.HDUList()
    numexts = len(data[0])

    for ext, header in zip(range(0,numexts,1),headerL):
        image.append(fits.ImageHDU(data[ext,:,:], header))

    image.writeto(filename, overwrite=True)
    print("\n" + filename + " was created!")
    return 0 

