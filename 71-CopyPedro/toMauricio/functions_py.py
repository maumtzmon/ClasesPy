import numpy as np
import math
from astropy.io import fits
import scipy.ndimage as ndimage
import glob
import copy
import pandas as pd
import os
import re
from scipy.optimize import curve_fit
from scipy import optimize
from dateutil.parser import parse
from scipy.stats import poisson, norm
import datetime
from matplotlib import colors
import argparse
import sys
import warnings
import skimage.measure as sk
from scipy.optimize import OptimizeWarning
from scipy.signal import find_peaks
import numpy.ma as ma
warnings.simplefilter("error", OptimizeWarning) # If the fit are not good get an error
warnings.simplefilter("error", RuntimeWarning) # for overflow in exponentials and ivalid values get an error

def no_sr(data,threshold=1.5,trem=10):
    h2 = data[:,9:]
    matriz2=np.zeros((h2.shape[0],h2.shape[1]-trem+1))
    for line in range(h2.shape[0]):
        for i in range(h2.shape[1]-trem+1):
            vagao=(h2[line,i:i+trem])

            if line != h2.shape[0]-1: vagaodebaixo=(h2[line+1,i:i+trem])
            else: vagaodebaixo=[]

            if line != 0: vagaodecima=(h2[line-1,i:i+trem])
            else: vagaodecima=[]

            baixo = 0
            cima = 0

            for alguembaixo in vagaodebaixo:
                if alguembaixo>=threshold: 
                    baixo=1
                    break
            for alguemcima in vagaodecima:
                if alguemcima>=threshold:
                    cima=1
                    break
                    
            for alguem in vagao:
                if alguem >=threshold and baixo ==0 and cima ==0: 
                    matriz2[line,i]=1
    #deleting dots
    for i in range(matriz2.shape[0]):
        for j in range(matriz2.shape[1]):
            if matriz2[i,j]==1:
                if j==0: #primeira coluna
                    if matriz2[i,j+1]==0 and matriz2[i,j+2]==0: matriz2[i,j]=0
                if j==1: #segunda coluna
                    if matriz2[i,j-1]==0 and matriz2[i,j+1]==0 and matriz2[i,j+2]==0: matriz2[i,j]=0
                if j==(matriz2.shape[1]-2): #penultima coluna
                    if matriz2[i,j-2]==0 and matriz2[i,j-1]==0 and matriz2[i,j+1]==0: matriz2[i,j]=0
                if j==(matriz2.shape[1]-1): #ultima coluna
                    if matriz2[i,j-1]==0 and matriz2[i,j-2]==0: matriz2[i,j]=0
                else:
                    if matriz2[i,j-2]==0 and matriz2[i,j-1]==0 and matriz2[i,j+1]==0: matriz2[i,j]=0
    #h2_nosr=np.zeros(h2.shape)
    dmask=np.zeros(data.shape)
    #h2_nosr[:,:]=h2[:,:]
    for i in range(matriz2.shape[0]):
        for j in range(matriz2.shape[1]):
            if matriz2[i,j]==1:
                    #h2_nosr[i,:]=-1e7
                    dmask[i,:]=128

    #return np.ma.masked_less(h2_nosr, -50000)
    return matriz2, dmask

def poisson_norm(x, mu, A, lamb, Nmax=5): #sigma parameter global 
    y = 0.
    for i in range(0, Nmax+1):
        y += (lamb**i)/float(math.factorial(i)) *np.exp(-0.5*((x-i-mu-lamb)/float(sigma))**2)
    return A*np.exp(-lamb)*y/(np.sqrt(2*np.pi*sigma**2))
      
def gaussian2(x,m,s,g,a1,a2):
    return a1*np.exp(-1/2*((x-m)/s)**2)+a2*np.exp(-1/2*((x-m-g)/s)**2)
    
def gauss(x,x0,s,A):
    return A*np.exp(-(x-x0)**2/(2*s**2))/np.sqrt(np.pi*2*s**2)

def ax(x,a,b):
    return a*x+b

def ax_nob(x,a):
    return a*x

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

def LocalCalib(data,extensions=1):
    if extensions==1:
        hist, bins = np.histogram( data[active_mask].flatten(), bins = np.arange(-400, 1000, 1) )
        x = (bins[1:] + bins[:-1])/2
        try:
            # popt_g, pcov = curve_fit( gaussian2, x, hist, p0=[ 0, 60, 780, 1e3, 1e2 ] )
            popt_g, pcov = curve_fit( gaussian2, x, hist, p0=[ 0, 60, 200, 1e3, 1e2 ] )
        except (RuntimeError,OptimizeWarning,RuntimeWarning):
            print(f"Error - gain fit failed")
        perr = np.sqrt(np.diag(pcov))
        gain = abs(popt_g[2])
        gain_err = perr[2]
        data=(data)/gain
        
    else:
        gain=[]
        gain_err=[]
        for i in range(extensions):
            hist, bins = np.histogram( data[i][active_mask].flatten(), bins = np.arange(-400, 1000, 1) )
            x = (bins[1:] + bins[:-1])/2
            try:
                #popt_g, pcov = curve_fit( gaussian2, x, hist, p0=[ 0, 150, 780, 1e3, 1e2 ] )
                popt_g, pcov = curve_fit( gaussian2, x, hist, p0=[ 0, 60, 200, 1e3, 1e2] )
                                                            # 1x2 p0=[ 0, 150, 770, 1e3, 1e2 ]
                                                            # 1x10 p0=[ 0, 150, 770, 1e2, 1e0 ] 
                perr = np.sqrt(np.diag(pcov))
                gain_list = abs(popt_g[2]-popt_g[0])
                gain_err_list = perr[2]

                data[i]=data[i]/gain_list
            except (RuntimeError,OptimizeWarning,RuntimeWarning):
                print(f"Error - gain fit failed")
                gain_list = -1000
                gain_err_list = -1000
            
            gain.append(gain_list)
            gain_err.append(gain_err_list)

    #return gain, gain_err, data
    return gain, gain_err, data, hist, x,popt_g #remove hist and x

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

def LocalSER(data,mask,sigma,readout_time,extensions=4):
    
    madata=ma.masked_array(data, mask)
    if extensions!=1:
        
        dc_list=[]
        dc_err_list=[]
        for k in range(extensions):
            global_mask=mask[k]
            Npix=global_mask.shape[0]*global_mask.shape[1]

            time_map=np.linspace(0,readout_time,num=Npix,endpoint=True).reshape(global_mask.shape[0],global_mask.shape[1])

            time_map_mask=ma.masked_where( global_mask.astype(bool), time_map)
            Nactive=ma.count(time_map_mask)
            exposure_time=ma.mean(time_map_mask)
#             print(Nactive,exposure_time)
            sigmaN=sigma[k]
            
            def poisson_normN(x, mu, A, lamb, Nmax=5): #sigma parameter global 
                y = 0.
                for i in range(0, Nmax+1):
                    y += (lamb**i)/float(math.factorial(i)) *np.exp(-0.5*((x-i-mu-lamb)/float(sigmaN))**2)
                return A*np.exp(-lamb)*y/(np.sqrt(2*np.pi*sigmaN**2))
            
            try:
                masked_hist, bins = np.histogram( ma.compressed(madata[k][convolution_mask]), np.arange(-0.5, 2.5, .01) )
                x = (bins[1:]+bins[:-1])/2
                
                popt, pcov = curve_fit( 
                    poisson_normN, 
                    x, 
                    masked_hist, 
                    p0=[-0.4, 1000, 0.05],
                )
                perr = np.sqrt(np.diag(pcov))
                dc = popt[2]/exposure_time # electron/pix/day
                dc_err = perr[2]/exposure_time # electron/pix/day                
            except (RuntimeError,OptimizeWarning,RuntimeWarning):
                print( f"Error - dc fit failed at" )
                (dc, dc_err)=(-1000,-1000)
            dc_list.append(dc)
            dc_err_list.append(dc_err)
        return (dc_list,dc_err_list)
    
def sum_intensity(region, intensities):
    return np.sum(intensities[region])

def mask_bleeding(data,direction,iterations,extensions=1,he_th=80):
    
    strcx=[[0,0,0],[0,1,1],[0,0,0]]
    strcy=[[0,0,0],[0,1,0],[0,1,0]]
    
    if extensions!=1: 
        mask_bleeding_array=[]
        HE_events_array=[]
        
        for i in range(extensions):
            catalog=ndimage.label(data[i]>4,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]
            rps=sk.regionprops(catalog,intensity_image=data[i],cache=False, extra_properties=[sum_intensity])
            energy=[r.sum_intensity for r in rps]
            ecce=[r.eccentricity for r in rps]

            hee_list=np.where((np.array(energy)>he_th) & (np.array(ecce)!=1))[0].tolist() # index of events that satisfy these conditions
            HE_events=np.zeros_like(catalog)
            for event in hee_list:
                [x,y]=np.where(catalog==event+1) #catalog index starts with 1 not 0
                for i in range(len(x)):
                    HE_events[x[i],y[i]]=1 #binary image to be dilated
                    
            if direction=='x':
                mask_bleeding=ndimage.morphology.binary_dilation(HE_events>0,structure=strcx,
                                                                 iterations=iterations)*1.0-(HE_events>0)*1.0
            if direction=='y':
                mask_bleeding=ndimage.morphology.binary_dilation(HE_events>0,structure=strcy,
                                                                 iterations=iterations)*1.0-(HE_events>0)*1.0
            if direction=='xy':
                mask_bleedingX=ndimage.morphology.binary_dilation(HE_events>0,structure=strcx,
                                                                 iterations=iterations[0])*1.0-(HE_events>0)*1.0
                mask_bleedingY=ndimage.morphology.binary_dilation(HE_events>0,structure=strcy,
                                                                 iterations=iterations[1])*1.0-(HE_events>0)*1.0
                mask_bleeding=mask_bleedingX+mask_bleedingY
            
            mask_bleeding_array.append(mask_bleeding)
            HE_events_array.append(HE_events)

            
    else:
        catalog=ndimage.label(data>4,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]
        rps=sk.regionprops(catalog,intensity_image=data,cache=False, extra_properties=[sum_intensity])
        energy=[r.sum_intensity for r in rps]
        ecce=[r.eccentricity for r in rps]

        hee_list=np.where((np.array(energy)>he_th) & (np.array(ecce)!=1))[0].tolist() # index of events that satisfy these conditions
        HE_events_array=np.zeros_like(catalog)
        for event in hee_list:
            [x,y]=np.where(catalog==event+1) #catalog index starts with 1 not 0
            for i in range(len(x)):
                HE_events_array[x[i],y[i]]=1 #binary image to be dilated

        if direction=='x':
            mask_bleeding_array=ndimage.morphology.binary_dilation(HE_events_array>0,structure=strcx,
                                                             iterations=iterations)*1.0-(HE_events_array>0)*1.0
        if direction=='y':
            mask_bleeding_array=ndimage.morphology.binary_dilation(HE_events_array>0,structure=strcy,
                                                             iterations=iterations)*1.0-(HE_events_array>0)*1.0
        if direction=='xy':
            mask_bleedingX=ndimage.morphology.binary_dilation(HE_events_array>0,structure=strcx,
                                                             iterations=iterations[0])*1.0-(HE_events_array>0)*1.0
            mask_bleedingY=ndimage.morphology.binary_dilation(HE_events_array>0,structure=strcy,
                                                             iterations=iterations[1])*1.0-(HE_events_array>0)*1.0
            mask_bleeding_array=mask_bleedingX+mask_bleedingY
            
    return np.array(mask_bleeding_array), np.array(HE_events_array)

def mask_SRE(data_masked,extensions=1,SRE_th=1.5):
    if extensions==1:
        pre_sre=ndimage.label(data_masked>SRE_th,structure=[[0,0,0],[1,1,1],[0,0,0]])[0]
        rps=sk.regionprops(pre_sre,cache=False)
        areas=[r.area for r in rps]

        SRE_events=np.zeros_like(pre_sre)
        
        for event in np.where(np.array(areas)>1)[0].tolist():
            [x,y]=np.where(pre_sre==event+1)
            for i in range(len(x)):
                SRE_events[x[i],y[i]]=1
        SRE_mask_array=np.zeros_like(SRE_events)
        for y in np.unique(np.where(SRE_events==1)[0]).tolist(): SRE_mask_array[y,:]=1
            
    else:
        SRE_mask_array=[]
        for i in range(extensions):
            pre_sre=ndimage.label(data_masked[i]>SRE_th,structure=[[0,0,0],[1,1,1],[0,0,0]])[0]
            rps=sk.regionprops(pre_sre,cache=False)
            areas=[r.area for r in rps]

            SRE_events=np.zeros_like(pre_sre)

            for event in np.where(np.array(areas)>1)[0].tolist():
                [x,y]=np.where(pre_sre==event+1)
                for i in range(len(x)):
                    SRE_events[x[i],y[i]]=1
                    
            SRE_mask=np.zeros_like(SRE_events)
            for y in np.unique(np.where(SRE_events==1)[0]).tolist(): SRE_mask[y,:]=1        
            SRE_mask_array.append(SRE_mask)
    return np.array(SRE_mask_array)

def GetRDtime(hdu_list):

    start=parse(hdu_list[0].header["DATESTART"]+"Z").timestamp()
    end=parse(hdu_list[0].header["DATEEND"]+"Z").timestamp()
    time=datetime.timedelta(seconds=end-start)
    readout_time=time.total_seconds()/86400
    return readout_time

def DataMminus(data2mask,mask):
    return data2mask-1e7*mask

def EventHalo_Mask(dataWevents,bleedingMask,extensions=4):
    if extensions!=1:
        event_mask=DataMminus(dataWevents,bleedingMask)
        event_halo_array=[]
        for i in range(extensions):
            event_halo_mask = ndimage.binary_dilation(
                    (event_mask[i]>4)|bleedingMask[i].astype(bool),
                    iterations = 10,
                    structure = ndimage.generate_binary_structure(rank=2, connectivity=2) # == [[1,1,1],[1,1,1],[1,1,1]]
                )
            event_halo_array.append(event_halo_mask)
        return np.array(event_halo_array)
    
    
global active_mask, overscan_mask, verticaloverscan_mask, convolution_mask, os_median_mask



osc=550 #163 microchip
# osc = 85

# active_mask = np.s_[:, 9:-(osc+5)] # == slice(None, None), slice(9, -151)
# overscan_mask = np.s_[5:-5, -(osc-5):-5] # == slice(None, None), slice(-149, None)
# verticaloverscan_mask = np.s_[:100,:]# remove median of first 100 rows
active_mask = np.s_[:, 0:osc] # == slice(None, None), slice(9, -151)
overscan_mask = np.s_[5:-5, -(osc-5):-5] # == slice(None, None), slice(-149, None)
verticaloverscan_mask = np.s_[:100,:]# remove median of first 100 rows
os_median_mask = np.s_[:, osc:]
# os_median_mask = np.s_[:, -(osc-5):-5]
convolution_mask = np.s_[5:-5, 15:-(osc+5)]

